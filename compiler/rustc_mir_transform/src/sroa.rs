use rustc_abi::{FieldIdx, VariantIdx};
use rustc_data_structures::flat_map_in_place::FlatMapInPlace;
use rustc_hir::LangItem;
use rustc_index::IndexVec;
use rustc_index::bit_set::{DenseBitSet, GrowableBitSet};
use rustc_middle::bug;
use rustc_middle::mir::visit::*;
use rustc_middle::mir::*;
use rustc_middle::ty::{self, Ty, TyCtxt};
use rustc_mir_dataflow::value_analysis::{excluded_locals, iter_fields};
use tracing::{debug, instrument};

use crate::patch::MirPatch;
use crate::ty::data_structures::HashMap;

pub(super) struct ScalarReplacementOfAggregates;
#[derive(Debug)]
struct EscapeVisitor {
    pub escaping: DenseBitSet<Local>,
    // Would swapping this for a IndexVec<Option<VariantIdx>> be better?
    pub variants: HashMap<Local, VariantIdx>,
}
impl<'tcx> Visitor<'tcx> for EscapeVisitor {
    fn visit_local(&mut self, local: Local, _: PlaceContext, _: Location) {
        self.escaping.insert(local);
    }
    #[track_caller]
    fn visit_place(&mut self, place: &Place<'tcx>, context: PlaceContext, location: Location) {
        if let &[PlaceElem::Downcast(_, variant_index), ..] = &place.projection[..] {
            // If this local was previously dowcast to another variant, then this local is not
            // eligible for SROA.
            if self
                .variants
                .insert(place.local, variant_index)
                .is_some_and(|old_idx| old_idx != variant_index)
            {
                eprintln!(
                    "Not apllying SRoA for {:?} because it is read as a different variant!",
                    place.local
                );
                self.escaping.insert(place.local);
            }
            eprintln!("{:?}", &place.projection[..]);
            return;
        }
        // Mirror the implementation in PreFlattenVisitor.
        if let &[PlaceElem::Field(..), ..] = &place.projection[..] {
            return;
        }

        self.super_place(place, context, location);
    }

    fn visit_assign(&mut self, lvalue: &Place<'tcx>, rvalue: &Rvalue<'tcx>, location: Location) {
        if lvalue.as_local().is_some() {
            match rvalue {
                // Handling the discirminant's of enums in SRoA is tricky, so it Discriminant is simply not supported yet.
                Rvalue::Discriminant(place) if let Some(local) = place.as_local() => {
                    self.escaping.insert(local);
                }
                // Aggregate assignments are expanded in run_pass.
                Rvalue::Aggregate(kind, _) => {
                    if let AggregateKind::Adt(_, variant_index, ..) = &**kind {
                        if self
                            .variants
                            .insert(lvalue.local, *variant_index)
                            .is_some_and(|old_idx| old_idx != *variant_index)
                        {
                            eprintln!(
                                "Not apllying SRoA for {:?} because it is initialized to a different variant!",
                                lvalue.local
                            );
                            self.escaping.insert(lvalue.local);
                        }
                    };
                    self.visit_rvalue(rvalue, location);
                    return;
                }
                Rvalue::Use(..) => {
                    self.visit_rvalue(rvalue, location);
                    return;
                }
                _ => {}
            }
        }
        self.super_assign(lvalue, rvalue, location)
    }

    fn visit_statement(&mut self, statement: &Statement<'tcx>, location: Location) {
        match &statement.kind {
            // Storage statements are expanded in run_pass.
            StatementKind::SetDiscriminant { place, variant_index }
                if let Some(local) = place.as_local() =>
            {
                if self
                    .variants
                    .insert(local, *variant_index)
                    .is_some_and(|old_idx| old_idx != *variant_index)
                {
                    eprintln!(
                        "Not apllying SRoA for {:?} because it's discriminant is set to a different variant!",
                        local
                    );
                    self.escaping.insert(local);
                }
            }
            StatementKind::StorageLive(..)
            | StatementKind::StorageDead(..)
            | StatementKind::Deinit(..) => return,
            _ => self.super_statement(statement, location),
        }
    }

    // We ignore anything that happens in debuginfo, since we expand it using
    // `VarDebugInfoFragment`.
    fn visit_var_debug_info(&mut self, _: &VarDebugInfo<'tcx>) {}
}
impl<'tcx> crate::MirPass<'tcx> for ScalarReplacementOfAggregates {
    fn is_enabled(&self, sess: &rustc_session::Session) -> bool {
        sess.mir_opt_level() >= 2
    }

    #[instrument(level = "debug", skip(self, tcx, body))]
    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        debug!(def_id = ?body.source.def_id());
        eprintln!("old:{body:?}");
        // Avoid query cycles (coroutines require optimized MIR for layout).
        if tcx.type_of(body.source.def_id()).instantiate_identity().is_coroutine() {
            return;
        }

        let mut excluded = excluded_locals(body);
        let typing_env = body.typing_env(tcx);
        loop {
            debug!(?excluded);
            let escaping = escaping_locals(tcx, typing_env, &excluded, body);
            eprintln!("{:?} {:?}", body.source, escaping);
            let replacements = compute_flattening(tcx, typing_env, body, escaping);
            eprintln!("replacements:{replacements:?}");
            debug!(?replacements);
            let all_dead_locals = replace_flattened_locals(tcx, body, replacements);

            if !all_dead_locals.is_empty() {
                excluded.union(&all_dead_locals);
                excluded = {
                    let mut growable = GrowableBitSet::from(excluded);
                    growable.ensure(body.local_decls.len());
                    growable.into()
                };
            } else {
                break;
            }
        }
        eprintln!("new:{body:?}");
    }

    fn is_required(&self) -> bool {
        false
    }
}

/// Identify all locals that are not eligible for SROA.
///
/// There are 4 cases:
/// - the aggregated local is used or passed to other code (function parameters and arguments);
/// - the locals is a union;
/// - the local is an enum, and more than one of its variants is accessed;
/// - the local's address is taken, and thus the relative addresses of the fields are observable to
///   client code.
fn escaping_locals<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    excluded: &DenseBitSet<Local>,
    body: &Body<'tcx>,
) -> EscapeVisitor {
    let is_excluded_ty = |ty: Ty<'tcx>| {
        if ty.is_union() {
            return true;
        }
        if let ty::Adt(def, _args) = ty.kind() {
            if def.repr().simd() {
                // Exclude #[repr(simd)] types so that they are not de-optimized into an array
                return true;
            }
            if tcx.is_lang_item(def.did(), LangItem::DynMetadata) {
                // codegen wants to see the `DynMetadata<T>`,
                // not the inner reference-to-opaque-type.
                return true;
            }
            if ty.is_enum() {
                return false;
            }
            for variant in def.variants() {
                if variant.fields.len() > 1 {
                    // If this has more than one field, it cannot be a wrapper that only provides a
                    // niche, so we do not want to automatically exclude it.
                    return false;
                }
            }
            let Ok(layout) = tcx.layout_of(typing_env.as_query_input(ty)) else {
                // We can't get the layout
                return true;
            };
            if layout.layout.largest_niche().is_some() {
                // This type has a niche
                return true;
            }
        }
        // Default for non-ADTs
        false
    };

    let mut escaping = DenseBitSet::new_empty(body.local_decls.len());
    escaping.insert_range(RETURN_PLACE..=Local::from_usize(body.arg_count));
    for (local, decl) in body.local_decls().iter_enumerated() {
        if excluded.contains(local) || is_excluded_ty(decl.ty) {
            eprintln!("{local:?} {decl:?} ");
            escaping.insert(local);
        }
    }
    eprintln!("init set:{escaping:?}");
    let mut visitor = EscapeVisitor { escaping, variants: Default::default() };
    visitor.visit_body(body);
    return visitor;
}

#[derive(Default, Debug)]
struct ReplacementMap<'tcx> {
    /// Pre-computed list of all "new" locals for each "old" local. This is used to expand storage
    /// and deinit statement and debuginfo.
    fragments: IndexVec<Local, Option<IndexVec<FieldIdx, Option<(Ty<'tcx>, Local)>>>>,
    variants: HashMap<Local, VariantIdx>,
}

impl<'tcx> ReplacementMap<'tcx> {
    fn replace_place(&self, tcx: TyCtxt<'tcx>, place: PlaceRef<'tcx>) -> Option<Place<'tcx>> {
        let &[PlaceElem::Field(f, _), ref rest @ ..] = place.projection else {
            let &[PlaceElem::Downcast(_, variant), PlaceElem::Field(f, _), ref rest @ ..] =
                place.projection
            else {
                return None;
            };
            let fields = self.fragments[place.local].as_ref()?;
            assert_eq!(variant, *self.variants.get(&place.local)?);
            let (_, new_local) = fields[f]?;
            return Some(Place { local: new_local, projection: tcx.mk_place_elems(rest) });
        };
        let fields = self.fragments[place.local].as_ref()?;
        let (_, new_local) = fields[f]?;
        Some(Place { local: new_local, projection: tcx.mk_place_elems(rest) })
    }

    fn place_fragments(
        &self,
        place: Place<'tcx>,
    ) -> Option<impl Iterator<Item = (FieldIdx, Ty<'tcx>, Local)>> {
        let local = place.as_local()?;
        eprintln!("local:{local:?}");
        let fields = self.fragments[local].as_ref()?;
        Some(fields.iter_enumerated().filter_map(|(field, &opt_ty_local)| {
            let (ty, local) = opt_ty_local?;
            Some((field, ty, local))
        }))
    }
}

/// Compute the replacement of flattened places into locals.
///
/// For each eligible place, we assign a new local to each accessed field.
/// The replacement will be done later in `ReplacementVisitor`.
fn compute_flattening<'tcx>(
    tcx: TyCtxt<'tcx>,
    typing_env: ty::TypingEnv<'tcx>,
    body: &mut Body<'tcx>,
    escaping: EscapeVisitor,
) -> ReplacementMap<'tcx> {
    let mut fragments = IndexVec::from_elem(None, &body.local_decls);

    for local in body.local_decls.indices() {
        if escaping.escaping.contains(local) {
            continue;
        }
        let decl = body.local_decls[local].clone();
        let ty = decl.ty;
        iter_fields(ty, tcx, typing_env, |variant, field, field_ty| {
            if variant.is_some() {
                // variants are currently not supported.
                //return;
            };
            let new_local =
                body.local_decls.push(LocalDecl { ty: field_ty, user_ty: None, ..decl.clone() });
            fragments.get_or_insert_with(local, IndexVec::new).insert(field, (field_ty, new_local));
        });
    }
    ReplacementMap { fragments, variants: escaping.variants }
}

/// Perform the replacement computed by `compute_flattening`.
fn replace_flattened_locals<'tcx>(
    tcx: TyCtxt<'tcx>,
    body: &mut Body<'tcx>,
    replacements: ReplacementMap<'tcx>,
) -> DenseBitSet<Local> {
    let mut all_dead_locals = DenseBitSet::new_empty(replacements.fragments.len());
    for (local, replacements) in replacements.fragments.iter_enumerated() {
        if replacements.is_some() {
            all_dead_locals.insert(local);
        }
    }
    debug!(?all_dead_locals);
    if all_dead_locals.is_empty() {
        return all_dead_locals;
    }

    let mut visitor = ReplacementVisitor {
        tcx,
        local_decls: &body.local_decls,
        replacements: &replacements,
        all_dead_locals,
        patch: MirPatch::new(body),
    };
    for (bb, data) in body.basic_blocks.as_mut_preserves_cfg().iter_enumerated_mut() {
        visitor.visit_basic_block_data(bb, data);
    }
    for scope in &mut body.source_scopes {
        visitor.visit_source_scope_data(scope);
    }
    for (index, annotation) in body.user_type_annotations.iter_enumerated_mut() {
        visitor.visit_user_type_annotation(index, annotation);
    }
    visitor.expand_var_debug_info(&mut body.var_debug_info);
    let ReplacementVisitor { patch, all_dead_locals, .. } = visitor;
    patch.apply(body);
    all_dead_locals
}

struct ReplacementVisitor<'tcx, 'll> {
    tcx: TyCtxt<'tcx>,
    /// This is only used to compute the type for `VarDebugInfoFragment`.
    local_decls: &'ll LocalDecls<'tcx>,
    /// Work to do.
    replacements: &'ll ReplacementMap<'tcx>,
    /// This is used to check that we are not leaving references to replaced locals behind.
    all_dead_locals: DenseBitSet<Local>,
    patch: MirPatch<'tcx>,
}

impl<'tcx> ReplacementVisitor<'tcx, '_> {
    #[instrument(level = "trace", skip(self))]
    fn expand_var_debug_info(&mut self, var_debug_info: &mut Vec<VarDebugInfo<'tcx>>) {
        var_debug_info.flat_map_in_place(|mut var_debug_info| {
            let place = match var_debug_info.value {
                VarDebugInfoContents::Const(_) => return vec![var_debug_info],
                VarDebugInfoContents::Place(ref mut place) => place,
            };

            if let Some(repl) = self.replacements.replace_place(self.tcx, place.as_ref()) {
                *place = repl;
                return vec![var_debug_info];
            }

            let Some(parts) = self.replacements.place_fragments(*place) else {
                return vec![var_debug_info];
            };

            let ty = place.ty(self.local_decls, self.tcx).ty;

            parts
                .map(|(field, field_ty, replacement_local)| {
                    let mut var_debug_info = var_debug_info.clone();
                    let composite = var_debug_info.composite.get_or_insert_with(|| {
                        Box::new(VarDebugInfoFragment { ty, projection: Vec::new() })
                    });
                    composite.projection.push(PlaceElem::Field(field, field_ty));

                    var_debug_info.value = VarDebugInfoContents::Place(replacement_local.into());
                    eprintln!("dg_ty:{ty:?} var_debug_info:{var_debug_info:?}");
                    var_debug_info
                })
                .collect()
        });
    }
}

impl<'tcx, 'll> MutVisitor<'tcx> for ReplacementVisitor<'tcx, 'll> {
    fn tcx(&self) -> TyCtxt<'tcx> {
        self.tcx
    }

    fn visit_place(&mut self, place: &mut Place<'tcx>, context: PlaceContext, location: Location) {
        if let Some(repl) = self.replacements.replace_place(self.tcx, place.as_ref()) {
            *place = repl
        } else {
            self.super_place(place, context, location)
        }
    }

    #[instrument(level = "trace", skip(self))]
    fn visit_statement(&mut self, statement: &mut Statement<'tcx>, location: Location) {
        match statement.kind {
            // Duplicate storage and deinit statements, as they pretty much apply to all fields.
            StatementKind::StorageLive(l) => {
                if let Some(final_locals) = self.replacements.place_fragments(l.into()) {
                    for (_, _, fl) in final_locals {
                        self.patch.add_statement(location, StatementKind::StorageLive(fl));
                    }
                    statement.make_nop();
                }
                return;
            }
            StatementKind::StorageDead(l) => {
                if let Some(final_locals) = self.replacements.place_fragments(l.into()) {
                    for (_, _, fl) in final_locals {
                        self.patch.add_statement(location, StatementKind::StorageDead(fl));
                    }
                    statement.make_nop();
                }
                return;
            }
            StatementKind::Deinit(box place) => {
                if let Some(final_locals) = self.replacements.place_fragments(place) {
                    for (_, _, fl) in final_locals {
                        self.patch
                            .add_statement(location, StatementKind::Deinit(Box::new(fl.into())));
                    }
                    statement.make_nop();
                    return;
                }
            }

            // We have `a = Struct { 0: x, 1: y, .. }`.
            // We replace it by
            // ```
            // a_0 = x
            // a_1 = y
            // ...
            // ```
            StatementKind::Assign(box (place, Rvalue::Aggregate(_, ref mut operands))) => {
                if let Some(local) = place.as_local()
                    && let Some(final_locals) = &self.replacements.fragments[local]
                {
                    // This is ok as we delete the statement later.
                    let operands = std::mem::take(operands);
                    for (&opt_ty_local, mut operand) in final_locals.iter().zip(operands) {
                        if let Some((_, new_local)) = opt_ty_local {
                            // Replace mentions of SROA'd locals that appear in the operand.
                            self.visit_operand(&mut operand, location);

                            let rvalue = Rvalue::Use(operand);
                            self.patch.add_statement(
                                location,
                                StatementKind::Assign(Box::new((new_local.into(), rvalue))),
                            );
                        }
                    }
                    statement.make_nop();
                    return;
                }
            }

            // We have `a = some constant`
            // We add the projections.
            // ```
            // a_0 = a.0
            // a_1 = a.1
            // ...
            // ```
            // ConstProp will pick up the pieces and replace them by actual constants.
            StatementKind::Assign(box (place, Rvalue::Use(Operand::Constant(_)))) => {
                if let Some(final_locals) = self.replacements.place_fragments(place) {
                    // Put the deaggregated statements *after* the original one.
                    let location = location.successor_within_block();
                    let adt_ty = place.ty(self.local_decls, self.tcx);
                    for (field, ty, new_local) in final_locals {
                        let rplace = if adt_ty.ty.is_enum() {
                            let variant = self.replacements.variants.get(&place.local).unwrap();
                            let adt = adt_ty.ty.ty_adt_def().unwrap();
                            let downcast = self.tcx.mk_place_downcast(place, adt, *variant);
                            self.tcx.mk_place_field(downcast, field, ty)
                        } else {
                            self.tcx.mk_place_field(place, field, ty)
                        };
                        let rvalue = Rvalue::Use(Operand::Move(rplace));
                        self.patch.add_statement(
                            location,
                            StatementKind::Assign(Box::new((new_local.into(), rvalue))),
                        );
                    }
                    // We still need `place.local` to exist, so don't make it nop.
                    return;
                }
            }

            // We have `a = move? place`
            // We replace it by
            // ```
            // a_0 = move? place.0
            // a_1 = move? place.1
            // ...
            // ```
            StatementKind::Assign(box (lhs, Rvalue::Use(ref op))) => {
                let (rplace, copy) = match *op {
                    Operand::Copy(rplace) => (rplace, true),
                    Operand::Move(rplace) => (rplace, false),
                    Operand::Constant(_) => bug!(),
                };
                let adt_ty = op.ty(self.local_decls, self.tcx);
                if let Some(final_locals) = self.replacements.place_fragments(lhs) {
                    for (field, ty, new_local) in final_locals {
                        let rplace = if adt_ty.is_enum() {
                            let variant = self.replacements.variants.get(&rplace.local).unwrap();
                            let adt = adt_ty.ty_adt_def().unwrap();
                            let downcast = self.tcx.mk_place_downcast(rplace, adt, *variant);
                            self.tcx.mk_place_field(downcast, field, ty)
                        } else {
                            self.tcx.mk_place_field(rplace, field, ty)
                        };
                        debug!(?rplace);
                        let rplace = self
                            .replacements
                            .replace_place(self.tcx, rplace.as_ref())
                            .unwrap_or(rplace);
                        debug!(?rplace);
                        let rvalue = if copy {
                            Rvalue::Use(Operand::Copy(rplace))
                        } else {
                            Rvalue::Use(Operand::Move(rplace))
                        };
                        self.patch.add_statement(
                            location,
                            StatementKind::Assign(Box::new((new_local.into(), rvalue))),
                        );
                    }
                    statement.make_nop();
                    return;
                }
                eprintln!(
                    "no replacement for {statement:?} {:?}",
                    self.replacements.place_fragments(lhs).map(|it| it.collect::<Vec<_>>())
                );
            }
            /*StatementKind::Assign(box (lhs, Rvalue::Discriminant(ref rhs)))
                if let Some(local) = rhs.as_local() =>
            {
                let discr = self.replacements.variants.get(&local).unwrap();
                let enum_ty = rhs.ty(self.local_decls, self.tcx).ty;

                let discr_value = enum_ty.discriminant_for_variant(self.tcx, *discr).unwrap();
                // Is this correct?
                let span = statement.source_info.span;
                // TODO: this is incorrect. `fully_monomorphized` should not be used in MIR opts.
                let env = TypingEnv::fully_monomorphized();
                let opr = ConstOperand{span,user_ty:None,const_:Const:: from_bits(self.tcx,discr_value.val, env, discr_value.ty)};
                statement.kind = StatementKind::Assign(Box::new((
                    lhs,
                    Rvalue::Use(Operand::Constant(Box::new(opr))),
                )));
                return;
            }*/
            _ => {}
        }
        eprintln!("{statement:?}");
        self.super_statement(statement, location)
    }

    fn visit_local(&mut self, local: &mut Local, _: PlaceContext, _: Location) {
        assert!(!self.all_dead_locals.contains(*local));
    }
}
