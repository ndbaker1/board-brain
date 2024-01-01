use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{loss, Optimizer, VarMap};

use crate::games::{get_move_interactive, BoardGame, Move, Player};

pub mod mlp;

pub const MODEL_FILE: &str = "checkpoint.safetensors";

pub trait Model {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

pub trait GameFactory<G: BoardGame> {
    fn new(&self) -> G;
}

const LEARNING_RATE: f64 = 0.1;

#[allow(dead_code)]
pub enum TrainingMode {
    Epoch(usize),
    TargetLoss(f32, Option<usize>),
}

struct TrainingSet {
    input: Tensor,
    output: Tensor,
}

struct TrainingSession {
    sets: Vec<TrainingSet>,
}

pub struct SelfPlayTrainer<'m, M: Model> {
    pub log: bool,
    pub mode: TrainingMode,
    pub model: &'m M,
}

impl<M: Model> SelfPlayTrainer<'_, M> {
    pub fn train(
        &mut self,
        game: &mut impl BoardGame,
        varmap: &VarMap,
        device: &Device,
    ) -> anyhow::Result<()> {
        let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
        let mut step_count = 0u32;

        let mut train_iter = || -> anyhow::Result<f32> {
            game.reset();
            let TrainingSession { sets: states } = self_play_session(game, self.model, device)?;

            let mut final_loss: Option<Tensor> = None;
            for TrainingSet { input, output } in states {
                let logits = self.model.forward(&input)?.squeeze(0)?;
                let loss = loss::mse(&logits, &output)?;
                sgd.backward_step(&loss)?;

                step_count += 1;
                if step_count.rem_euclid(150) == 0 {
                    println!(
                        "Step: {step_count:5} | Loss: {:8.5} | LR: {:1.5}",
                        loss.to_scalar::<f32>()?,
                        sgd.learning_rate()
                    );
                }

                final_loss.replace(loss);
            }

            let Some(loss) = final_loss else {
                anyhow::bail!("no loss for iteration");
            };

            Ok(loss.to_scalar::<f32>()?)
        };

        match self.mode {
            TrainingMode::TargetLoss(target_loss, checkpoint_freq) => {
                let mut iter = 0usize;
                while train_iter()? > target_loss {
                    iter += 1;
                    if checkpoint_freq.is_some_and(|freq| iter.rem_euclid(freq) == 0) {
                        varmap.save(MODEL_FILE)?;
                    }
                }
            }
            TrainingMode::Epoch(epochs) => {
                for _ in 1..=epochs {
                    train_iter()?;
                }
            }
        }

        Ok(())
    }
}

pub fn play(game: &mut impl BoardGame, model: &impl Model, device: &Device) -> anyhow::Result<()> {
    let winner = loop {
        // bot goes first
        if let Some(winner) = game.play(get_move_model(game, model, device)?)? {
            break winner;
        }
        println!("{:?}", game);
        if let Some(winner) = game.play(get_move_interactive()?)? {
            break winner;
        }
    };
    println!("🥳 player [{:?}] Won!", winner);
    println!("{}", game);
    Ok(())
}

fn self_play_session<G: BoardGame>(
    game: &mut G,
    model: &impl Model,
    device: &Device,
) -> anyhow::Result<TrainingSession> {
    let mut logs = Vec::new();
    let winner = loop {
        let ai_move = get_move_model(game, model, device)?;
        // log events of the game
        logs.push((
            game.get_turn(),
            game.get_board().into_iter().collect::<Vec<_>>(),
            ai_move,
        ));
        if let Some(winner) = game.play(ai_move)? {
            break winner;
        }
    };

    let mut states = Vec::new();
    // only use turns where the player was the winner or there was a draw
    for (_, game_state, player_move) in logs
        .iter()
        .filter(|(player, ..)| winner == Player::None || *player == winner)
    {
        states.push(TrainingSet {
            input: Tensor::from_iter(game_state.iter().cloned().map(Into::into), &device)?
                .to_dtype(DType::F32)?,
            output: Tensor::from_iter(G::create_answer_tensor(*player_move, logs.len()), &device)?
                .to_dtype(DType::F32)?,
        })
    }

    Ok(TrainingSession { sets: states })
}

fn get_move_model<G: BoardGame>(
    game: &G,
    model: &impl Model,
    device: &Device,
) -> anyhow::Result<Move> {
    let mut max_val = f32::NEG_INFINITY;
    let mut max_pos = Move(0, 0);

    let state = Tensor::from_iter(game.get_board().into_iter().map(Into::into), &device)?
        .to_dtype(DType::F32)?;

    let play_distribution = model.forward(&state)?;
    for items in play_distribution.to_vec2::<f32>()? {
        for (index, value) in items.iter().enumerate() {
            if G::is_space_empty(&game.get_board()[index]) && *value > max_val {
                max_pos = Move::from_1d(index, G::BOARD_COLUMNS);
                max_val = *value;
            }
        }
    }

    Ok(max_pos)
}
