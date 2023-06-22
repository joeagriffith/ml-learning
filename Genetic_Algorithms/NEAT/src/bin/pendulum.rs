use gym_rs::{ActionType, PendulumEnv, GymEnv, GifRender};
use neat_from_scratch::{Neat, Genome, FeedForwardNetwork};

fn main() {
    let mut neat = Neat::new_fully_connected();
    let champ = neat.train(pendulum_env, 6000.0);
    render_champion(&champ);
    champ.print();
}

 
pub fn pendulum_env(genome: &Genome) -> f64 {
    let mut env = PendulumEnv::default();
    env.seed(0);
    let mut state:Vec<f64> = env.reset();
    let mut end = false;
    let mut total_reward = 0.0;

    let mut nn = FeedForwardNetwork::new(genome);
    let mut steps = 0;

    while !end {
        if steps >= 300 {
            break;
        }

        let output = nn.activate(state);
        let action = ActionType::Continuous(vec![output[0] * 4.0 - 2.0]);
        let (s, reward, done, _info) = env.step(action);
        end = done;
        state = s;
        total_reward += reward;
        steps += 1;
    }
    total_reward += 5000.0;
    if total_reward < 0.0 { total_reward = 0.0; }
    total_reward
}

fn render_champion(champ: &Genome) {
    println!("Rendering Champ");

    let mut env = PendulumEnv::default();
    let mut render = GifRender::new(540, 540, "img/pendulum_champion.gif", 50).unwrap();

    let mut state = env.reset();
    let mut end = false;
    let mut steps = 0;

    let mut nn = FeedForwardNetwork::new(champ);

    while !end {
        println!("step {steps}");
        if steps > 300 {
            break;
        }
        let output = nn.activate(state);
        let action = ActionType::Continuous(vec![output[0] * 4.0 - 2.0]);

        let (s, _reward, done, _info) = env.step(action);
        end = done;
        state = s;
        steps += 1;

        env.render(&mut render);
    }
}