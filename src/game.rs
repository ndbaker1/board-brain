use std::fmt::Display;

pub const BOARD_COLS: usize = 5;
pub const BOARD_ROWS: usize = 8;
pub const fn space_count() -> usize {
    BOARD_COLS * BOARD_ROWS
}

/// x and Y coordinates
#[derive(Copy, Clone, Debug)]
pub struct Move(pub usize, pub usize);

pub type Board = [Player; space_count()];

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum Player {
    None,
    One,
    Two,
}

impl Move {
    pub fn to_1d(&self) -> usize {
        self.0 + self.1 * BOARD_COLS
    }
    pub fn from_1d(i: usize) -> Self {
        Self(i.rem_euclid(BOARD_COLS), i / BOARD_COLS)
    }
}
impl Player {
    fn flip(&self) -> Self {
        match self {
            Self::One => Self::Two,
            Self::Two => Self::One,
            _ => *self,
        }
    }
}

pub struct Connect4Spin {
    pub current_state: Board,
    pub turn: Player,
}

impl Connect4Spin {
    pub fn new() -> Self {
        Self {
            current_state: [Player::None; space_count()],
            turn: Player::One,
        }
    }

    fn check_winner(&self) -> Option<Player> {
        for player in [Player::One, Player::Two] {
            // Check for a horizontal win
            for row in 0..BOARD_ROWS {
                for col in 0..BOARD_COLS - 3 {
                    if self.current_state[Move(col, row).to_1d()] == player
                        && self.current_state[Move(col + 1, row).to_1d()] == player
                        && self.current_state[Move(col + 2, row).to_1d()] == player
                        && self.current_state[Move(col + 3, row).to_1d()] == player
                    {
                        return Some(player);
                    }
                }
            }

            // Check for a vertical win
            for row in 0..BOARD_ROWS - 3 {
                for col in 0..BOARD_COLS {
                    if self.current_state[Move(col, row).to_1d()] == player
                        && self.current_state[Move(col, row + 1).to_1d()] == player
                        && self.current_state[Move(col, row + 2).to_1d()] == player
                        && self.current_state[Move(col, row + 3).to_1d()] == player
                    {
                        return Some(player);
                    }
                }
            }

            // Check for a diagonal win (top-left to bottom-right)
            for row in 0..BOARD_ROWS - 3 {
                for col in 0..BOARD_COLS - 3 {
                    if self.current_state[Move(col, row).to_1d()] == player
                        && self.current_state[Move(col + 1, row + 1).to_1d()] == player
                        && self.current_state[Move(col + 2, row + 2).to_1d()] == player
                        && self.current_state[Move(col + 3, row + 3).to_1d()] == player
                    {
                        return Some(player);
                    }
                }
            }

            // Check for a diagonal win (bottom-left to top-right)
            for row in 3..BOARD_ROWS {
                for col in 0..BOARD_COLS - 3 {
                    if self.current_state[Move(col, row).to_1d()] == player
                        && self.current_state[Move(col + 1, row - 1).to_1d()] == player
                        && self.current_state[Move(col + 2, row - 2).to_1d()] == player
                        && self.current_state[Move(col + 3, row - 3).to_1d()] == player
                    {
                        return Some(player);
                    }
                }
            }
        }

        None
    }

    pub fn play(&mut self, pos: Move) -> anyhow::Result<Option<Player>> {
        if pos.0 < BOARD_COLS && pos.1 < BOARD_ROWS {
            // temporary hack to avoid stalling the game
            if !matches!(self.current_state[pos.to_1d()], Player::None) {
                eprintln!("{:?} was taken.", pos);

                let mut placed = false;
                for v in &mut self.current_state {
                    if matches!(v, Player::None) {
                        *v = self.turn;
                        placed = true;
                        break;
                    }
                }
                if !placed {
                    return Ok(Some(Player::None));
                }
            } else {
                // set the player's move on the board
                self.current_state[pos.to_1d()] = self.turn;
            }
            // make it the other player's turn
            self.turn = self.turn.flip();
            // randomly flip the column after playing
            if rand::random() {
                let temp: Vec<_> = (0..BOARD_ROWS)
                    .map(|row| self.current_state[Move(pos.0, row).to_1d()])
                    .collect();
                for (index, v) in temp.into_iter().rev().enumerate() {
                    self.current_state[Move(pos.0, index).to_1d()] = v;
                }
            }
            // if there is win, then return it
            return Ok(self.check_winner());
        } else {
            Err(anyhow::anyhow!("{:?} is out of bounds.", pos))
        }
    }
}

impl Display for Connect4Spin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..BOARD_ROWS {
            write!(f, "{}", y + 1)?;
            for x in 0..BOARD_COLS {
                let icon = match self.current_state[Move(x, y).to_1d()] {
                    Player::One => "🔵",
                    Player::Two => "🔴",
                    Player::None => "🟩",
                };
                write!(f, "{icon}")?;
            }
            writeln!(f)?;
        }

        write!(f, " ")?;
        for x in 0..BOARD_COLS {
            write!(f, "{:2}", char::from_u32(x as u32 + 97).unwrap())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::{Connect4Spin, Move};

    #[test]
    fn test() {
        let mut game = Connect4Spin::new();
        let player1_moves = [Move(0, 0)];
        let player2_moves = [Move(0, 1)];
        for player_move in player1_moves
            .chunks(1)
            .zip(player2_moves.chunks(1))
            .flat_map(|(a, b)| a.into_iter().chain(b))
        {
            match game.play(*player_move) {
                Ok(Some(_)) => break,
                Ok(None) => continue,
                Err(err) => panic!("{err}"),
            }
        }
    }
}
