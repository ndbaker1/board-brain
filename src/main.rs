#![feature(iter_array_chunks)]
#![feature(iterator_try_collect)]

use std::io::Write;

use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{loss, Linear, Module, Optimizer, VarBuilder, VarMap};
use game::Board;

use crate::game::{BOARD_COLS, BOARD_ROWS};

mod game;

const STATE: usize = game::space_count();
const CHOICES: usize = game::space_count();
const EPOCHS: usize = 500;
const LEARNING_RATE: f64 = 0.05;

struct Model {
    ln1: Linear,
    ln2: Linear,
    ln3: Linear,
}

impl Model {
    fn new(vs: VarBuilder) -> anyhow::Result<Self> {
        const LAYER1_OUT_SIZE: usize = 36;
        const LAYER2_OUT_SIZE: usize = 20;

        let ln1 = candle_nn::linear(STATE, LAYER1_OUT_SIZE, vs.pp("ln1"))?;
        let ln2 = candle_nn::linear(LAYER1_OUT_SIZE, LAYER2_OUT_SIZE, vs.pp("ln2"))?;
        let ln3 = candle_nn::linear(LAYER2_OUT_SIZE, CHOICES, vs.pp("ln3"))?;
        Ok(Self { ln1, ln2, ln3 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.ln1.forward(&xs.reshape((1, STATE))?)?;
        let xs = xs.relu()?;
        let xs = self.ln2.forward(&xs)?;
        let xs = xs.relu()?;
        self.ln3.forward(&xs)
    }
}

struct TrainingSet {
    game_states: Vec<Tensor>,
    move_states: Vec<Tensor>,
}

pub fn main() -> anyhow::Result<()> {
    let device = Device::cuda_if_available(0)?;
    println!("🧠 Training model..");
    let model = train(&device, EPOCHS)?;
    println!("🎮 Creating game with model..");
    play(&model, &device)
}

fn play(model: &Model, device: &Device) -> anyhow::Result<()> {
    let mut game = game::Connect4Spin::new();

    let winner = loop {
        // bot goes first
        let state = Tensor::from_iter(
            game.current_state.to_vec().into_iter().map(|v| v as u8),
            &device,
        )?
        .to_dtype(DType::F32)?;

        let mut max_val = 0.0f32;
        let mut max_pos = game::Move(0, 0);

        let play_distribution = model.forward(&state)?;
        for items in play_distribution.to_vec2::<f32>()? {
            for (index, value) in items.iter().enumerate() {
                if *value > max_val {
                    max_pos = game::Move::from_1d(index);
                    max_val = *value;
                }
            }
        }

        if let Some(winner) = game.play(max_pos)? {
            break winner;
        }
        if let Some(winner) = game.play(interactive_move(&game)?)? {
            break winner;
        }
    };

    println!("🥳 Player [{:?}] Won!", winner);
    println!("{}", game);

    Ok(())
}

fn train(dev: &Device, epochs: usize) -> anyhow::Result<Model> {
    let varmap = VarMap::new();
    let vs = VarBuilder::from_varmap(&varmap, DType::F32, dev);
    let model = Model::new(vs.clone())?;
    let mut sgd = candle_nn::SGD::new(varmap.all_vars(), LEARNING_RATE)?;
    let mut step_count = 0u32;
    for epoch in 1..=epochs {
        let TrainingSet {
            game_states,
            move_states,
        } = self_play_training_data(&model, &dev)?;

        for (game_state, move_state) in game_states.into_iter().zip(move_states) {
            let logits = model.forward(&game_state)?.squeeze(0)?;
            let loss = loss::mse(&logits, &move_state)?;
            sgd.backward_step(&loss)?;

            step_count += 1;
            if step_count.rem_euclid(100) == 0 {
                println!("Epoch: {epoch:5} | Loss: {:8.5}", loss.to_scalar::<f32>()?,);
            }
        }
    }
    Ok(model)
}

fn interactive_move(game: &game::Connect4Spin) -> anyhow::Result<game::Move> {
    println!("{}", game);

    fn request() -> anyhow::Result<game::Move> {
        print!("where would you like to go [ex. d3]> ");
        std::io::stdout().flush()?;
        let mut buffer = String::new();
        std::io::stdin().read_line(&mut buffer)?;
        let [x, y] = buffer
            .trim()
            .chars()
            .array_chunks::<2>()
            .next()
            .ok_or_else(|| anyhow::anyhow!(""))?;
        let (x, y) = (x as usize - 97, y as usize - (48 + 1));
        if x >= BOARD_COLS || y >= BOARD_ROWS {
            anyhow::bail!("out of bounds");
        }
        Ok(game::Move(x, y))
    }

    loop {
        match request() {
            Err(err) => println!("Bad input, try again. [{err}]"),
            m @ Ok(_) => return m,
        }
    }
}

fn create_answer_tensor(pmove: &game::Move, state: &Board) -> [f32; game::space_count()] {
    let mut board: [f32; game::space_count()] = [0.0; game::space_count()];
    for (index, value) in board.iter_mut().enumerate() {
        if index == pmove.to_1d() {
            *value = 1.0
        } else if state[index] == game::Player::None {
            *value = 0.0
        }
    }
    board
}

fn self_play_training_data(model: &Model, device: &Device) -> anyhow::Result<TrainingSet> {
    let mut game = game::Connect4Spin::new();

    let mut log = Vec::new();

    let mut max_val = 0.0f32;
    let mut max_pos = game::Move(0, 0);

    let winner = loop {
        let state = Tensor::from_iter(
            game.current_state.to_vec().into_iter().map(|v| v as u8),
            device,
        )?
        .to_dtype(DType::F32)?;

        let play_distribution = model.forward(&state)?;
        for items in play_distribution.to_vec2::<f32>()? {
            for (index, value) in items.iter().enumerate() {
                if *value > max_val {
                    max_pos = game::Move::from_1d(index);
                    max_val = *value;
                }
            }
        }

        log.push((game.turn, game.current_state, max_pos));

        match game.play(max_pos)? {
            Some(winner) => break winner,
            None => continue,
        }
    };

    #[cfg(debug_assertions)]
    eprintln!("Game stats> turns [{}], winner [{:?}]", log.len(), winner);

    let winning_data = log
        .into_iter()
        .filter(|(player, ..)| winner == game::Player::None || *player == winner);

    Ok(TrainingSet {
        game_states: winning_data
            .clone()
            .map(|(_, game_state, _)| {
                Tensor::from_iter(game_state.to_vec().into_iter().map(|v| v as u8), &device)?
                    .to_dtype(DType::F32)
            })
            .try_collect()?,
        move_states: winning_data
            .clone()
            .map(|(_, game_state, player_move)| {
                Tensor::from_iter(
                    create_answer_tensor(&player_move, &game_state)
                        .to_vec()
                        .into_iter(),
                    &device,
                )?
                .to_dtype(DType::F32)
            })
            .try_collect()?,
    })
}
