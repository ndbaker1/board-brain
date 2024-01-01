#![feature(iter_array_chunks)]
#![feature(iterator_try_collect)]

use std::{panic, path::Path};

use candle_core::{DType, Device};
use candle_nn::{VarBuilder, VarMap};
use cliclack::intro;
use games::BoardGame;
use models::mlp;

use crate::{
    games::connect4spin,
    models::{play, SelfPlayTrainer, TrainingMode, MODEL_FILE},
};

mod games;
mod models;

pub trait Named {
    /// A name to use to references this definition
    const NAME: &'static str;
}

pub fn main() -> anyhow::Result<()> {
    intro("🧠 Board-Brain")?;

    let mut game = match cliclack::select("Select game")
        .item(connect4spin::Game::NAME, connect4spin::Game::NAME, "")
        .interact()?
    {
        connect4spin::Game::NAME => connect4spin::Game::new(),
        _ => panic!(),
    };

    let mut spinner = cliclack::spinner();
    spinner.start("loading components");
    let device = Device::cuda_if_available(0)?;
    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    spinner.stop("✅ components loaded");

    let model = match cliclack::select("Select model")
        .item(mlp::MLP::NAME, "Multi-Layer Perceptron", "")
        .interact()?
    {
        mlp::MLP::NAME => mlp::MLP::new(vs, 40, 40)?,
        _ => panic!(),
    };

    let model_path = format!("{}.{}.{}", game, model, MODEL_FILE);

    if Path::new(&model_path).exists() {
        cliclack::log::info("📦 loading existing model parameters..")?;
        varmap.load(&model_path)?;
    }

    let mut trainer = SelfPlayTrainer {
        model: &model,
        log: true,
        mode: TrainingMode::TargetLoss(0.001, Some(100)),
    };
    cliclack::log::remark("🧠 training model..")?;
    trainer.train(&mut game, &varmap, &device)?;

    cliclack::log::success("💾 saving model..")?;
    varmap.save(&model_path)?;

    cliclack::outro("✅ model ready")?;

    println!("🎮 creating {} game with model..", game);
    game.reset();
    play(&mut game, &model, &device)?;
    Ok(())
}
