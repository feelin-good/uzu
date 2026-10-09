use super::{gemm::GemmPlan, gemv::GemvSpecialization};

#[derive(Debug)]
pub enum MatmulDispatch {
    Gemv(GemvSpecialization),
    Gemm(GemmPlan),
}
