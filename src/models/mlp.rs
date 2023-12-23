use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::{Model, c4spin};

const STATE: usize = c4spin::space_count();

pub struct MLP {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}

impl MLP {
    pub fn new(vs: VarBuilder) -> anyhow::Result<Self> {
        const LAYER1_OUT_SIZE: usize = 36;
        const LAYER2_OUT_SIZE: usize = 20;

        let ln1 = candle_nn::linear(STATE, LAYER1_OUT_SIZE, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(LAYER1_OUT_SIZE, LAYER2_OUT_SIZE, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(LAYER2_OUT_SIZE, STATE, vs.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3 })
    }
}

impl Model for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(&xs.reshape((1, STATE))?)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        self.ln3.forward(&xs)
    }
}
