use std::{fmt::{Display, Debug}, io::Write, ops::Index};

pub trait BoardGame: Display + Debug {
    const BOARD_COLUMNS: usize;
    const BOARD_ROWS: usize;

    type Space: Clone + Into<u8>;
    type Board: Index<usize, Output = Self::Space> + IntoIterator<Item = Self::Space>;

    fn new() -> Self;
    fn reset(&mut self);
    /// play a move on the board game
    fn play(&mut self, pmove: Move) -> anyhow::Result<Option<Player>>;
    /// checks if a move is out of bounds
    fn is_move_oob(&self, pmove: Move) -> bool;
    fn get_turn(&self) -> Turn;
    fn get_board(&self) -> Self::Board;
    fn is_space_empty(space: &Self::Space) -> bool;

    fn create_answer_tensor(pmove: Move, turn_count: usize) -> impl IntoIterator<Item = f32> {
        (0..(Self::BOARD_COLUMNS * Self::BOARD_ROWS))
            .map(|s| Move::from_1d(s, Self::BOARD_COLUMNS))
            .map(move |m| {
                if m == pmove {
                    // rule of this, that the game can be ended in 7 turns max
                    20.0 / turn_count as f32
                } else {
                    0.0
                }
            })
    }
}

type Turn = Player;

/// (x,y) coordinate pairs
#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub struct Move(pub usize, pub usize);
impl Move {
    pub fn to_1d(&self, columns: usize) -> usize {
        self.0 + self.1 * columns
    }
    pub fn from_1d(i: usize, columns: usize) -> Self {
        Self(i.rem_euclid(columns), i / columns)
    }
}

#[derive(Copy, Clone, PartialEq, Debug, Eq, Hash)]
pub enum Player {
    /// represents an unassigned space or a tie between players
    None,
    /// represents the player who has the first action
    One,
    /// represents the player who has the second action
    Two,
}
impl Player {
    pub fn flip(&self) -> Self {
        match self {
            Self::One => Self::Two,
            Self::Two => Self::One,
            _ => *self,
        }
    }
}
impl Into<u8> for Player {
    fn into(self) -> u8 {
        match self {
            Self::None => 0,
            Self::One => 1,
            Self::Two => 2,
        }
    }
}

/// Help for getting a move from the user
pub fn get_move_interactive() -> anyhow::Result<Move> {
    let request = move || -> anyhow::Result<Move> {
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
        let pos = Move(x, y);
        Ok(pos)
    };

    loop {
        match request() {
            Err(err) => println!("Bad input, try again. [{err}]"),
            Ok(m) => break Ok(m),
        }
    }
}

// implementations...

pub mod connect4spin;
