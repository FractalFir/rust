use std::ffi::{OsStr, OsString};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::thread;

use anyhow::{anyhow, Context, Result};
use dunce::canonicalize;
use path_macro::path;
use xshell::{cmd, Shell};

pub fn miri_dir() -> std::io::Result<PathBuf> {
    const MIRI_SCRIPT_ROOT_DIR: &str = env!("CARGO_MANIFEST_DIR");
    Ok(canonicalize(MIRI_SCRIPT_ROOT_DIR)?.parent().unwrap().into())
}

/// Queries the active toolchain for the Miri dir.
pub fn active_toolchain() -> Result<String> {
    let sh = Shell::new()?;
    sh.change_dir(miri_dir()?);
    let stdout = cmd!(sh, "rustup show active-toolchain").read()?;
    Ok(stdout.split_whitespace().next().context("Could not obtain active Rust toolchain")?.into())
}

pub fn flagsplit(flags: &str) -> Vec<String> {
    // This code is taken from `RUSTFLAGS` handling in cargo.
    flags.split(' ').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect()
}

pub fn arg_flag_value(
    args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    flag: &str,
) -> Option<OsString> {
    let mut args = args.into_iter();
    while let Some(arg) = args.next() {
        let arg = arg.as_ref();
        if arg == "--" {
            return None;
        }
        let Some(arg) = arg.to_str() else {
            // Skip non-UTF-8 arguments.
            continue;
        };
        if arg == flag {
            // Next one is the value.
            return Some(args.next()?.as_ref().to_owned());
        } else if let Some(val) = arg.strip_prefix(flag).and_then(|s| s.strip_prefix("=")) {
            return Some(val.to_owned().into());
        }
    }
    None
}

/// Some extra state we track for building Miri, such as the right RUSTFLAGS.
pub struct MiriEnv {
    /// miri_dir is the root of the miri repository checkout we are working in.
    pub miri_dir: PathBuf,
    /// active_toolchain is passed as `+toolchain` argument to cargo/rustc invocations.
    pub toolchain: String,
    /// Extra flags to pass to cargo.
    pub cargo_extra_flags: Vec<String>,
    /// The rustc sysroot
    pub sysroot: PathBuf,
    /// The shell we use.
    pub sh: Shell,
}

impl MiriEnv {
    pub fn new() -> Result<Self> {
        let toolchain = active_toolchain()?;
        let sh = Shell::new()?; // we are preserving the current_dir on this one, so paths resolve properly!
        let miri_dir = miri_dir()?;

        let sysroot = cmd!(sh, "rustc +{toolchain} --print sysroot").read()?.into();
        let target_output = cmd!(sh, "rustc +{toolchain} --version --verbose").read()?;
        let rustc_meta = rustc_version::version_meta_for(&target_output)?;
        let libdir = path!(sysroot / "lib" / "rustlib" / rustc_meta.host / "lib");

        // Determine some toolchain properties
        if !libdir.exists() {
            println!("Something went wrong determining the library dir.");
            println!("I got {} but that does not exist.", libdir.display());
            println!("Please report a bug at https://github.com/rust-lang/miri/issues.");
            std::process::exit(2);
        }
        // Share target dir between `miri` and `cargo-miri`.
        let target_dir = std::env::var_os("CARGO_TARGET_DIR")
            .unwrap_or_else(|| path!(miri_dir / "target").into());
        sh.set_var("CARGO_TARGET_DIR", target_dir);

        // We configure dev builds to not be unusably slow.
        let devel_opt_level =
            std::env::var_os("CARGO_PROFILE_DEV_OPT_LEVEL").unwrap_or_else(|| "2".into());
        sh.set_var("CARGO_PROFILE_DEV_OPT_LEVEL", devel_opt_level);

        // Compute rustflags.
        let rustflags = {
            let mut flags = OsString::new();
            // We set the rpath so that Miri finds the private rustc libraries it needs.
            flags.push("-C link-args=-Wl,-rpath,");
            flags.push(libdir);
            // Enable rustc-specific lints (ignored without `-Zunstable-options`).
            flags.push(
                " -Zunstable-options -Wrustc::internal -Wrust_2018_idioms -Wunused_lifetimes",
            );
            // Add user-defined flags.
            if let Some(value) = std::env::var_os("RUSTFLAGS") {
                flags.push(" ");
                flags.push(value);
            }
            flags
        };
        sh.set_var("RUSTFLAGS", rustflags);

        // Get extra flags for cargo.
        let cargo_extra_flags = std::env::var("CARGO_EXTRA_FLAGS").unwrap_or_default();
        let cargo_extra_flags = flagsplit(&cargo_extra_flags);

        Ok(MiriEnv { miri_dir, toolchain, sh, sysroot, cargo_extra_flags })
    }

    pub fn install_to_sysroot(
        &self,
        path: impl AsRef<OsStr>,
        args: impl IntoIterator<Item = impl AsRef<OsStr>>,
    ) -> Result<()> {
        let MiriEnv { sysroot, toolchain, cargo_extra_flags, .. } = self;
        // Install binaries to the miri toolchain's `sysroot` so they do not interact with other toolchains.
        cmd!(self.sh, "cargo +{toolchain} install {cargo_extra_flags...} --path {path} --force --root {sysroot} {args...}").run()?;
        Ok(())
    }

    pub fn build(
        &self,
        manifest_path: impl AsRef<OsStr>,
        args: &[OsString],
        quiet: bool,
    ) -> Result<()> {
        let MiriEnv { toolchain, cargo_extra_flags, .. } = self;
        let quiet_flag = if quiet { Some("--quiet") } else { None };
        // We build the tests as well, (a) to avoid having rebuilds when building the tests later
        // and (b) to have more parallelism during the build of Miri and its tests.
        let mut cmd = cmd!(
            self.sh,
            "cargo +{toolchain} build --bins --tests {cargo_extra_flags...} --manifest-path {manifest_path} {quiet_flag...} {args...}"
        );
        cmd.set_quiet(quiet);
        cmd.run()?;
        Ok(())
    }

    pub fn check(&self, manifest_path: impl AsRef<OsStr>, args: &[OsString]) -> Result<()> {
        let MiriEnv { toolchain, cargo_extra_flags, .. } = self;
        cmd!(self.sh, "cargo +{toolchain} check {cargo_extra_flags...} --manifest-path {manifest_path} --all-targets {args...}")
            .run()?;
        Ok(())
    }

    pub fn clippy(&self, manifest_path: impl AsRef<OsStr>, args: &[OsString]) -> Result<()> {
        let MiriEnv { toolchain, cargo_extra_flags, .. } = self;
        cmd!(self.sh, "cargo +{toolchain} clippy {cargo_extra_flags...} --manifest-path {manifest_path} --all-targets {args...}")
            .run()?;
        Ok(())
    }

    pub fn test(&self, manifest_path: impl AsRef<OsStr>, args: &[OsString]) -> Result<()> {
        let MiriEnv { toolchain, cargo_extra_flags, .. } = self;
        cmd!(
            self.sh,
            "cargo +{toolchain} test {cargo_extra_flags...} --manifest-path {manifest_path} {args...}"
        )
        .run()?;
        Ok(())
    }

    /// Receives an iterator of files.
    /// Will format each file with the miri rustfmt config.
    /// Does not recursively format modules.
    pub fn format_files(
        &self,
        files: impl Iterator<Item = Result<PathBuf, walkdir::Error>>,
        toolchain: &str,
        config_path: &Path,
        flags: &[OsString],
    ) -> anyhow::Result<()> {
        use itertools::Itertools;

        let mut first = true;

        // Format in batches as not all our files fit into Windows' command argument limit.
        for batch in &files.chunks(256) {
            // Build base command.
            let mut cmd = cmd!(
                self.sh,
                "rustfmt +{toolchain} --edition=2021 --config-path {config_path} --unstable-features --skip-children {flags...}"
            );
            if first {
                // Log an abbreviating command, and only once.
                eprintln!("$ {cmd} ...");
                first = false;
            }
            // Add files.
            for file in batch {
                // Make it a relative path so that on platforms with extremely tight argument
                // limits (like Windows), we become immune to someone cloning the repo
                // 50 directories deep.
                let file = file?;
                let file = file.strip_prefix(&self.miri_dir)?;
                cmd = cmd.arg(file);
            }

            // Run rustfmt.
            // We want our own error message, repeating the command is too much.
            cmd.quiet().run().map_err(|_| anyhow!("`rustfmt` failed"))?;
        }

        Ok(())
    }

    /// Run the given closure many times in parallel with access to the shell, once for each value in the `range`.
    pub fn run_many_times(
        &self,
        range: Range<u32>,
        run: impl Fn(&Shell, u32) -> Result<()> + Sync,
    ) -> Result<()> {
        // `next` is atomic so threads can concurrently fetch their next value to run.
        let next = AtomicU32::new(range.start);
        let end = range.end; // exclusive!
        let failed = AtomicBool::new(false);
        thread::scope(|s| {
            let mut handles = Vec::new();
            // Spawn one worker per core.
            for _ in 0..thread::available_parallelism()?.get() {
                // Create a copy of the shell for this thread.
                let local_shell = self.sh.clone();
                let handle = s.spawn(|| -> Result<()> {
                    let local_shell = local_shell; // move the copy into this thread.
                    // Each worker thread keeps asking for numbers until we're all done.
                    loop {
                        let cur = next.fetch_add(1, Ordering::Relaxed);
                        if cur >= end {
                            // We hit the upper limit and are done.
                            break;
                        }
                        // Run the command with this seed.
                        run(&local_shell, cur).map_err(|err| {
                            // If we failed, tell everyone about this.
                            failed.store(true, Ordering::Relaxed);
                            err
                        })?;
                        // Check if some other command failed (in which case we'll stop as well).
                        if failed.load(Ordering::Relaxed) {
                            return Ok(());
                        }
                    }
                    Ok(())
                });
                handles.push(handle);
            }
            // Wait for all workers to be done.
            for handle in handles {
                handle.join().unwrap()?;
            }
            // If all workers succeeded, we can't have failed.
            assert!(!failed.load(Ordering::Relaxed));
            Ok(())
        })
    }
}
