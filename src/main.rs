#![feature(iter_array_chunks)]
#![feature(iterator_try_collect)]

use std::{collections::HashMap, io::Write, path::Path};

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};

mod c4spin;
mod models;

const MODEL_FILE: &str = "checkpoint.safetensors";
const LEARNING_RATE: f64 = 0.1;

trait Model {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

#[allow(dead_code)]
struct TrainingSession {
    states: Vec<(Tensor, Tensor)>,
    logs: Vec<(c4spin::Player, c4spin::Board, c4spin::Move)>,
    winner: c4spin::Player,
}

pub fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;
    let mut varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, &device);
    let model = models::mlp::MLP::new(vs)?;
    if Path::new(MODEL_FILE).exists() {
        println!("📦 Loading vars..");
        varmap.load(MODEL_FILE)?;
    }
    println!("🧠 Training model..");
    train(&model, &varmap, &device)?;
    println!("💾 Saving model..");
    varmap.save(MODEL_FILE)?;
    println!("🎮 Creating game with model..");
    play(&model, &device)
}

fn play(model: &impl Model, device: &Device) -> anyhow::Result<()> {
    let mut game = c4spin::Game::new();
    let winner = loop {
        // bot goes first
        if let Some(winner) = game.play(get_move_model(&game, model, device)?)? {
            break winner;
        }
        if let Some(winner) = game.play(get_move_interactive(&game)?)? {
            break winner;
        }
    };
    println!("🥳 Player [{:?}] Won!", winner);
    println!("{}", game);
    Ok(())
}

fn train(model: &impl Model, varmap: &VarMap, device: &Device) -> anyhow::Result<()> {
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;

    let mut step_count = 0u32;
    let mut winners = HashMap::new();
    let mut first_moves = HashMap::new();

    loop {
        let TrainingSession {
            states,
            winner,
            logs,
        } = self_play_session(model, device)?;

        #[cfg(debug_assertions)]
        for ele in &logs {
            println!(
                "{}",
                c4spin::Game {
                    current_state: ele.1,
                    turn: ele.0
                }
            )
        }

        winners.entry(winner).and_modify(|v| *v += 1).or_insert(1);
        first_moves
            .entry(logs[0].2)
            .and_modify(|v| *v += 1)
            .or_insert(1);

        let mut final_loss: Option<Tensor> = None;
        for (game_state, move_state) in states {
            let logits = model.forward(&game_state)?.squeeze(0)?;
            let loss = loss::mse(&logits, &move_state)?;
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

        if let Some(loss) = final_loss {
            if loss.to_scalar::<f32>()? < 0.0001 {
                break;
            }
        }
    }

    println!("wins by turn: {:#?}", winners);
    println!("first move frequency: {:#?}", first_moves);

    Ok(())
}

fn get_move_model(
    game: &c4spin::Game,
    model: &impl Model,
    device: &Device,
) -> anyhow::Result<c4spin::Move> {
    let mut max_val = f32::NEG_INFINITY;
    let mut max_pos = c4spin::Move(0, 0);

    let state = Tensor::from_iter(
        game.current_state.to_vec().into_iter().map(|v| v as u8),
        &device,
    )?
    .to_dtype(DType::F32)?;

    let play_distribution = model.forward(&state)?;
    for items in play_distribution.to_vec2::<f32>()? {
        for (index, value) in items.iter().enumerate() {
            if game.current_state[index] == c4spin::Player::None && *value > max_val {
                max_pos = c4spin::Move::from_1d(index);
                max_val = *value;
            }
        }
    }

    Ok(max_pos)
}

fn get_move_interactive(game: &c4spin::Game) -> anyhow::Result<c4spin::Move> {
    println!("{}", game);

    let request = move || -> anyhow::Result<c4spin::Move> {
        print!("where would you like to go [ex. d3]> ");
        std::io::stdout().flush()?;
        let mut buffer = String::new();
        std::io::stdin().read_line(&mut buffer)?;
        // parse the character pairs as two coordinate values
        let [x, y] = buffer
            .trim()
            .chars()
            .array_chunks::<2>()
            .next()
            .ok_or_else(|| anyhow::anyhow!("not formatted correctly."))?;
        // remove the ascii base of the characters
        let (x, y) = (x as usize - 97, y as usize - (48 + 1));
        let pos = c4spin::Move(x, y);
        // check bounds
        if game.current_state.get(pos.to_1d()).is_none() {
            anyhow::bail!("out of bounds");
        }
        Ok(pos)
    };

    loop {
        match request() {
            Err(err) => println!("Bad input, try again. [{err}]"),
            m @ Ok(_) => return m,
        }
    }
}

fn create_answer_tensor(
    pmove: &c4spin::Move,
    state: &c4spin::Board,
    turn_count: usize,
) -> [f32; c4spin::space_count()] {
    let mut board: [f32; c4spin::space_count()] = [0.0; c4spin::space_count()];
    for (index, value) in board.iter_mut().enumerate() {
        if index == pmove.to_1d() {
            // rule of this, that the game can be ended in 7 turns max
            *value = 7.0 / turn_count as f32;
        } else if state[index] == c4spin::Player::None {
            *value = 0.0
        }
    }
    board
}

fn self_play_session(model: &impl Model, device: &Device) -> anyhow::Result<TrainingSession> {
    let mut game = c4spin::Game::new();
    let mut logs = Vec::new();
    let winner = loop {
        let ai_move = get_move_model(&game, model, device)?;
        // log events of the game
        logs.push((game.turn, game.current_state, ai_move));
        if let Some(winner) = game.play(ai_move)? {
            break winner;
        }
    };

    let mut states = Vec::new();
    // only use turns where the player was the winner or there was a draw
    for (_, game_state, player_move) in logs
        .iter()
        .filter(|(player, ..)| winner == c4spin::Player::None || *player == winner)
    {
        states.push((
            Tensor::from_iter(game_state.to_vec().into_iter().map(|v| v as u8), &device)?
                .to_dtype(DType::F32)?,
            Tensor::from_iter(
                create_answer_tensor(&player_move, &game_state, logs.len())
                    .to_vec()
                    .into_iter(),
                &device,
            )?
            .to_dtype(DType::F32)?,
        ))
    }

    Ok(TrainingSession {
        states,
        logs,
        winner,
    })
}
