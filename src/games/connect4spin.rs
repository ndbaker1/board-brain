use std::fmt::{Display, Debug};

use crate::Named;

use super::{BoardGame, Move, Player};

pub struct Game {
    pub current_state: [Player; Self::space_count()],
    pub turn: Player,
}

impl Named for Game {
    const NAME: &'static str = "connect-4-spin";
}

impl BoardGame for Game {
    const BOARD_COLUMNS: usize = 5;
    const BOARD_ROWS: usize = 8;

    type Space = Player;
    type Board = [Self::Space; Self::space_count()];

    fn new() -> Self {
        Self {
            current_state: [Player::None; Self::space_count()],
            turn: Player::One,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    fn is_space_empty(space: &Self::Space) -> bool {
        *space == Player::None
    }

    fn get_board(&self) -> Self::Board {
        self.current_state
    }

    fn get_turn(&self) -> super::Turn {
        self.turn
    }

    fn is_move_oob(&self, pmove: Move) -> bool {
        self.current_state
            .get(pmove.to_1d(Self::BOARD_COLUMNS))
            .is_none()
    }

    fn play(&mut self, pos: Move) -> anyhow::Result<Option<Player>> {
        // bounds check
        if self.is_move_oob(pos) {
            anyhow::bail!("{:?} is out of bounds", pos);
        }
        // temporary hack to avoid stalling the game
        if let Some(p @ (Player::One | Player::Two)) =
            self.current_state.get(pos.to_1d(Self::BOARD_COLUMNS))
        {
            anyhow::bail!("{:?} already belongs to {:?}", pos, p);
        }
        // set the player's move on the board
        self.current_state[pos.to_1d(Self::BOARD_COLUMNS)] = self.turn;
        // make it the other player's turn
        self.turn = self.turn.flip();
        // randomly flip the column after playing
        // TODO: use some extra logic based on the weight of the chips to control the flip chance
        if rand::random() {
            // create a temporary copy of the column
            let temp: Vec<_> = (0..Self::BOARD_ROWS)
                .map(|row| self.current_state[Move(pos.0, row).to_1d(Self::BOARD_COLUMNS)])
                .collect();
            // apply the reveresed copy onto the column
            for (index, v) in temp.into_iter().rev().enumerate() {
                self.current_state[Move(pos.0, index).to_1d(Self::BOARD_COLUMNS)] = v;
            }
        }
        // check if there is winner or a tie
        Ok(self.check_winner())
    }
}

impl Game {
    pub const fn space_count() -> usize {
        Self::BOARD_COLUMNS * Self::BOARD_ROWS
    }

    fn check_winner(&self) -> Option<Player> {
        for player in [Player::One, Player::Two] {
            // check for a horizontal win
            for row in 0..Self::BOARD_ROWS {
                for col in 0..Self::BOARD_COLUMNS - 3 {
                    if self.current_state[Move(col, row).to_1d(Self::BOARD_COLUMNS)] == player
                        && self.current_state[Move(col + 1, row).to_1d(Self::BOARD_COLUMNS)]
                            == player
                        && self.current_state[Move(col + 2, row).to_1d(Self::BOARD_COLUMNS)]
                            == player
                        && self.current_state[Move(col + 3, row).to_1d(Self::BOARD_COLUMNS)]
                            == player
                    {
                        return Some(player);
                    }
                }
            }

            // check for a vertical win
            for row in 0..Self::BOARD_ROWS - 3 {
                for col in 0..Self::BOARD_COLUMNS {
                    if self.current_state[Move(col, row).to_1d(Self::BOARD_COLUMNS)] == player
                        && self.current_state[Move(col, row + 1).to_1d(Self::BOARD_COLUMNS)]
                            == player
                        && self.current_state[Move(col, row + 2).to_1d(Self::BOARD_COLUMNS)]
                            == player
                        && self.current_state[Move(col, row + 3).to_1d(Self::BOARD_COLUMNS)]
                            == player
                    {
                        return Some(player);
                    }
                }
            }

            // check for a diagonal win (top-left to bottom-right)
            for row in 0..Self::BOARD_ROWS - 3 {
                for col in 0..Self::BOARD_COLUMNS - 3 {
                    if self.current_state[Move(col, row).to_1d(Self::BOARD_COLUMNS)] == player
                        && self.current_state[Move(col + 1, row + 1).to_1d(Self::BOARD_COLUMNS)]
                            == player
                        && self.current_state[Move(col + 2, row + 2).to_1d(Self::BOARD_COLUMNS)]
                            == player
                        && self.current_state[Move(col + 3, row + 3).to_1d(Self::BOARD_COLUMNS)]
                            == player
                    {
                        return Some(player);
                    }
                }
            }

            // check for a diagonal win (bottom-left to top-right)
            for row in 3..Self::BOARD_ROWS {
                for col in 0..Self::BOARD_COLUMNS - 3 {
                    if self.current_state[Move(col, row).to_1d(Self::BOARD_COLUMNS)] == player
                        && self.current_state[Move(col + 1, row - 1).to_1d(Self::BOARD_COLUMNS)]
                            == player
                        && self.current_state[Move(col + 2, row - 2).to_1d(Self::BOARD_COLUMNS)]
                            == player
                        && self.current_state[Move(col + 3, row - 3).to_1d(Self::BOARD_COLUMNS)]
                            == player
                    {
                        return Some(player);
                    }
                }
            }
        }

        // check for a tie
        self.current_state
            .iter()
            .all(|space| *space != Player::None)
            .then_some(Player::None)
    }
}

impl Display for Game {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", Self::NAME)
    }
}

impl Debug for Game {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for y in 0..Self::BOARD_ROWS {
            write!(f, "{}", y + 1)?;
            for x in 0..Self::BOARD_COLUMNS {
                let icon = match self.current_state[Move(x, y).to_1d(Self::BOARD_COLUMNS)] {
                    Player::One => "🔵",
                    Player::Two => "🔴",
                    Player::None => "🟩",
                };
                write!(f, "{icon}")?;
            }
            writeln!(f)?;
        }

        write!(f, " ")?;
        for x in 0..Self::BOARD_COLUMNS {
            // unicode characters are 2 wide
            write!(f, "{:2}", char::from_u32(x as u32 + 97).unwrap())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use crate::games::BoardGame;

    use super::{Game, Move};

    #[test]
    fn test() {
        let mut game = Game::new();
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
