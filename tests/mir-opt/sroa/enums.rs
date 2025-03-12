//@ compile-flags: -Cpanic=abort
//@ no-prefer-dynamic
/*
#[inline(never)]
fn option(a:u32)->u32{
    let a = Some(a); 
    match a{
        Some(a)=>a,
        _=>panic!(),
    }
}
enum Many{
    Float(f32),
    Double(f64),
    UInt(u32),
}
#[inline(never)]
fn many_inactive_variants(a:u32)->u32{
    let many = Many::UInt(a);
    match many{
        Many::UInt(a) => a,
        Many::Double(a) => a as u32,
        _=>todo!(),
    }
}
#[inline(never)]
fn many_possible_variants(a:u32)->u32{
    let many = if a == 0 {
          Many::Double(a as f64)
    }else{
          Many::UInt(a)
    };
    match many{
        Many::UInt(a) => a,
        _=>todo!(),
    }
}
*/
#[inline(never)]
pub unsafe fn unwrap_unchecked<T>(slf: Option<T>) -> T {
    match slf{
        Some(a)=>a,
        a=> std::hint::unreachable_unchecked(),
    }
}
fn main() {
    // CHECK-LABEL: fn main(
    //option(8);
    //many_inactive_variants(8);
    //many_possible_variants(8);
    unsafe{unwrap_unchecked(Some(8_u32))};
}
// EMIT_MIR enums.s.ScalarReplacementOfAggregates.mir
// EMIT_MIR enums.option.ScalarReplacementOfAggregates.mir
