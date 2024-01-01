use std::fmt::Display;

use candle_core::{Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::Named;

use super::Model;

pub struct MLP {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
    input_size: usize,
}

impl MLP {
    pub fn new(vs: VarBuilder, input_size: usize, output_size: usize) -> anyhow::Result<Self> {
        const LAYER1_OUT_SIZE: usize = 36;
        const LAYER2_OUT_SIZE: usize = 20;

        let ln1 = candle_nn::linear(input_size, LAYER1_OUT_SIZE, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(LAYER1_OUT_SIZE, LAYER2_OUT_SIZE, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(LAYER2_OUT_SIZE, output_size, vs.pp("ln3"))?;

        Ok(Self {
            ln1,
            ln2,
            ln3,
            input_size,
        })
    }
}

impl Display for MLP {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Self::NAME)
    }
}

impl Named for MLP {
    const NAME: &'static str = "mlp";
}

impl Model for MLP {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(&xs.reshape((1, self.input_size))?)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        self.ln3.forward(&xs)
    }
}
