use gym_rs::{ActionType, CartPoleEnv, GymEnv, GifRender};
use neat_from_scratch::{Neat, Genome, FeedForwardNetwork};

fn main() {
    let mut neat = Neat::new_fully_connected();
    let champ = neat.train(cartpole_env, 10000.0);
    render_champion(&champ);
    champ.print();
}

 
pub fn cartpole_env(genome: &Genome) -> f64 {
    let mut env = CartPoleEnv::default();
    let mut state:Vec<f64> = env.reset();
    let mut end = false;
    let mut total_reward = 0.0;

    let mut nn = FeedForwardNetwork::new(genome);

    while !end {
        if total_reward >= 10500.0 {
            break;
        }

        let output = nn.activate(state);
        let action: ActionType = if output[0] < 0.5 {
            ActionType::Discrete(0)
        } else {
            ActionType::Discrete(1)
        };
        let (s, reward, done, _info) = env.step(action);
        end = done;
        state = s;
        total_reward += reward;
    }
    total_reward
}

fn render_champion(champ: &Genome) {
    println!("Rendering Champ");

    let mut env = CartPoleEnv::default();
    let mut render = GifRender::new(540, 540, "cart_pole_champion.gif", 20).unwrap();

    let mut state = env.reset();
    let mut end = false;
    let mut steps = 0;

    let mut nn = FeedForwardNetwork::new(champ);

    while !end {
        println!("step {steps}");
        if steps > 300 {
            break;
        }
        println!("activating nn");
        let output = nn.activate(state);
        let action: ActionType = if output[0] < 0.5 {
            ActionType::Discrete(0)
        } else {
            ActionType::Discrete(1)
        };

        println!("env.step()");
        let (s, _reward, done, _info) = env.step(action);
        end = done;
        state = s;
        steps += 1;

        println!("env.render()");
        env.render(&mut render);
    }
}