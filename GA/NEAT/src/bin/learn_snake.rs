use neat_from_scratch::{Neat, Genome, FeedForwardNetwork, softmax};
use snake::*;

fn main() {
    let mut neat = Neat::new_fully_connected();
    let snake_genome = neat.train(snake_env, 749.0);
    snake_genome.print();
}

pub fn snake_env(genome: &Genome) -> f64 {
    let mut action:[usize;5] = [1, 0, 0, 0, 0];
    let mut game = SnakeGame::new(PlayType::ByMoveNoRender);

    let mut playing = true;
    let mut game_turns = 0.0;
    let mut game_score = 0.0;
    let mut stagnant_turns = 0;

    let mut neural_network = FeedForwardNetwork::new(genome);

    while playing {
        game_score = match game.make_action(action) {
            GameState::Playing((score, input)) => {
                stagnant_turns += 1;
                if score > game_score {
                    stagnant_turns = 0;
                }
                if stagnant_turns > 50 {
                    playing = false;
                }
                let input = Vec::from(input);
                let output = neural_network.activate(input);
                // action = softmax(output);

                game_turns += 1.0;
                score
            }
            GameState::GameOver(score) => {
                playing = false;
                score
            }
            GameState::Won(score) => {
                println!("Genome Won Snake!");
                playing = false;
                score
            }
        }
    }
    game_turns + (game_score * 100.0)
}