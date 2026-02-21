"""
Hardware Design Optimization using Deep Q-Network (DQN)

This script replaces the LLM agent with a DQN reinforcement learning agent
for intelligent hardware design space exploration.

Usage:
    python main_dqn.py --mode train --episodes 50
    python main_dqn.py --mode evaluate --load checkpoints/best_agent.pt
"""

from reinforcement_learning.training.dqn_agent import DQNAgent
from tools.run_yosys import synthesize
from tools.simulate import simulate
from tools.results_reporter import generate_all_reports
import math
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Disable simulations (optional - synthesis works perfectly without them)
os.environ.setdefault('RUN_SIMULATION', 'false')

# =============================================================================
# DESIGN CONSTRAINTS (Real-world optimization requirements)
# =============================================================================
MAX_AREA_CELLS = 1500      # Maximum total cells allowed
MIN_THROUGHPUT = 2          # Minimum ops/cycle required
MAX_FLIP_FLOPS = 400        # Maximum flip-flops allowed
CONSTRAINT_PENALTY = 10000  # Penalty for violating constraints

# =============================================================================
# OPTIMIZATION OBJECTIVE FUNCTION
# =============================================================================

def calculate_objective(params, metrics):
    """
    Calculate objective function for design optimization with constraints.
    
    Lower score = better design
    
    Metrics:
    - Area cost: Total cells (want to minimize)
    - Performance: Throughput = PAR operations per cycle (want to maximize)
    - Efficiency: Area per unit throughput
    
    Constraints (with penalties):
    - Maximum area (total cells)
    - Minimum throughput
    - Maximum flip-flops
    """
    par = params["PAR"]
    buffer_depth = params.get("BUFFER_DEPTH", 1024)
    
    total_cells = metrics.get('total_cells', float('inf'))
    flip_flops = metrics.get('flip_flops', 0)
    logic_cells = metrics.get('logic_cells', 0)
    
    if total_cells is None or total_cells == 0:
        return float('inf')
    
    # Performance metric: Effective throughput
    # Higher PAR = more parallel operations = better throughput
    throughput = par
    
    # Area-efficiency metric: cells per unit of throughput
    area_efficiency = total_cells / throughput
    
    # Base objective: Balance area and efficiency
    area_weight = 1.0
    efficiency_weight = 0.5
    objective = (area_weight * total_cells) + (efficiency_weight * area_efficiency)
    
    # Apply constraint penalties
    penalty = 0
    constraints_violated = []
    
    if total_cells > MAX_AREA_CELLS:
        penalty += CONSTRAINT_PENALTY * (1 + (total_cells - MAX_AREA_CELLS) / MAX_AREA_CELLS)
        constraints_violated.append(f"Area={total_cells} > {MAX_AREA_CELLS}")
    
    if par < MIN_THROUGHPUT:
        penalty += CONSTRAINT_PENALTY
        constraints_violated.append(f"Throughput={par} < {MIN_THROUGHPUT}")
    
    if flip_flops > MAX_FLIP_FLOPS:
        penalty += CONSTRAINT_PENALTY * 0.5  # Softer penalty for FFs
        constraints_violated.append(f"FFs={flip_flops} > {MAX_FLIP_FLOPS}")
    
    # Store constraint violations in metrics for reporting
    metrics['constraints_violated'] = constraints_violated
    metrics['constraint_penalty'] = penalty
    
    return objective + penalty


def evaluate_design(params, iteration, debug=False):
    """
    Evaluate a hardware design by generating RTL and synthesizing.
    
    Returns: (objective, metrics)
    """
    par = params["PAR"]
    buffer_depth = params.get("BUFFER_DEPTH", 1024)
    
    # Calculate buffer address width
    addr_width = int(math.ceil(math.log2(buffer_depth)))

    # Generate RTL dynamically with proposed parameters
    rtl = f"""
module reduce_sum #(
    parameter PAR = {par},
    parameter BUFFER_DEPTH = {buffer_depth}
) (
    input clk,
    input rst,
    input [31:0] in_data,
    input in_valid,
    output reg [31:0] out_data,
    output reg out_valid
);

reg [31:0] acc [0:PAR-1];
reg [{addr_width-1}:0] count;
integer i;

reg [31:0] final_sum;

always @(posedge clk) begin
    if (rst) begin
        for (i = 0; i < PAR; i = i + 1)
            acc[i] <= 0;
        count <= 0;
        out_valid <= 0;
    end
    else if (in_valid) begin
        for (i = 0; i < PAR; i = i + 1)
            acc[i] <= acc[i] + in_data + i;

        count <= count + 1;

        if (count == BUFFER_DEPTH - 1) begin
            final_sum = 0;
            for (i = 0; i < PAR; i = i + 1)
                final_sum = final_sum + acc[i];

            out_data <= final_sum;
            out_valid <= 1;
            count <= 0;
        end
    end
end

endmodule
"""

    with open("../../rtl/tmp.v", "w") as f:
        f.write(rtl)

    # Synthesize and collect metrics
    area, log, metrics = synthesize("../../rtl/tmp.v", debug=debug)

    # Add derived metrics
    metrics["area"] = area
    metrics["throughput"] = par
    metrics["area_per_throughput"] = area / par if par > 0 else float('inf')
    
    # Run functional simulation (if simulator available)
    run_simulation = os.environ.get('RUN_SIMULATION', 'true').lower() == 'true'
    if run_simulation:
        sim_success, sim_metrics, sim_log = simulate("../../rtl/tmp.v", params)
        metrics.update(sim_metrics)
        if not sim_success and debug:
            print(f"  ⚠️  Simulation: FAILED")
            if sim_log and len(sim_log) > 0:
                print(f"     Error: {sim_log[:200]}")
    
    # Calculate objective function
    objective = calculate_objective(params, metrics)
    metrics["objective"] = objective
    
    return objective, metrics


def run_training(episodes=50, iterations_per_episode=20, agent=None):
    """
    Train DQN agent to find optimal hardware designs.
    
    Args:
        episodes: Number of training episodes
        iterations_per_episode: Design evaluations per episode
        agent: DQNAgent instance (creates new if None)
    """
    
    if agent is None:
        agent = DQNAgent(
            state_dim=16,
            lr=0.001,
            gamma=0.95,
            epsilon_start=1.0,
            epsilon_end=0.05,
            epsilon_decay=0.995,
            batch_size=32,
            target_update_freq=10
        )
    
    print("\n" + "="*70)
    print(" 🎓 DQN TRAINING MODE - HARDWARE DESIGN OPTIMIZATION")
    print("="*70)
    print(f"\n📋 Training Configuration:")
    print(f"   Episodes: {episodes}")
    print(f"   Iterations per Episode: {iterations_per_episode}")
    print(f"   Total Designs to Explore: {episodes * iterations_per_episode}")
    print(f"\n⚖️  Design Constraints:")
    print(f"   • Max Area:       {MAX_AREA_CELLS} cells")
    print(f"   • Min Throughput: {MIN_THROUGHPUT} ops/cycle")
    print(f"   • Max Flip-Flops: {MAX_FLIP_FLOPS}")
    print("\n" + "="*70)
    
    # Track all-time best design
    global_best_objective = float('inf')
    global_best_design = None
    all_history = []
    
    # Training statistics
    episode_rewards = []
    episode_best_objectives = []
    
    for episode in range(episodes):
        print(f"\n{'='*70}")
        print(f"📦 EPISODE {episode + 1}/{episodes}")
        print(f"{'='*70}")
        print(f"   Epsilon (exploration): {agent.epsilon:.3f}")
        
        episode_history = []
        episode_reward = 0
        episode_best = float('inf')
        
        for iteration in range(iterations_per_episode):
            # Get current state
            state = agent.encode_state(all_history)
            
            # Agent selects action (design parameters)
            action_idx, params = agent.select_action(state, evaluation=False)
            
            # Evaluate design
            debug_mode = (episode == 0 and iteration == 0)
            objective, metrics = evaluate_design(params, iteration, debug=debug_mode)
            
            # Compute reward
            reward = agent.compute_reward(objective, metrics, all_history)
            episode_reward += reward
            
            # Get next state
            all_history.append((params, metrics))
            episode_history.append((params, metrics))
            next_state = agent.encode_state(all_history)
            
            # Store transition
            done = (iteration == iterations_per_episode - 1)
            agent.store_transition(state, action_idx, reward, next_state, done)
            
            # Train agent
            loss = agent.train_step()
            
            # Track episode best
            if objective < episode_best:
                episode_best = objective
            
            # Track global best
            if objective < global_best_objective:
                global_best_objective = objective
                global_best_design = (params.copy(), metrics.copy())
                print(f"\n   🎉 NEW GLOBAL BEST! Objective: {objective:.1f}")
            
            # Print iteration summary (compact)
            if iteration % 5 == 0 or iteration == iterations_per_episode - 1:
                par = params['PAR']
                bd = params['BUFFER_DEPTH']
                cells = metrics.get('total_cells', 0) or 0  # Handle None
                loss_val = f"{loss:.4f}" if loss else "0.0000"
                print(f"   Iter {iteration+1:2d}: PAR={par:2d}, BD={bd:4d}, "
                      f"Cells={cells:4d}, Obj={objective:6.1f}, Reward={reward:5.1f}, "
                      f"Loss={loss_val}")
        
        # Episode summary
        episode_rewards.append(episode_reward)
        episode_best_objectives.append(episode_best)
        
        print(f"\n   Episode Summary:")
        print(f"      Total Reward: {episode_reward:.2f}")
        print(f"      Best Objective: {episode_best:.1f}")
        print(f"      Designs Explored: {len(episode_history)}")
        print(f"      Memory Size: {len(agent.memory)}")
        
        # Update target network periodically
        if (episode + 1) % agent.target_update_freq == 0:
            agent.update_target_network()
            print(f"   🔄 Target network updated")
        
        # Decay exploration
        agent.decay_epsilon()
        
        # Save checkpoint every 10 episodes
        if (episode + 1) % 10 == 0:
            agent.save(f'reinforcement_learning/checkpoints/dqn_episode_{episode+1}.pt')
    
    # Final save
    agent.save('reinforcement_learning/checkpoints/dqn_final.pt')
    
    # Training complete summary
    print("\n\n" + "="*70)
    print(" 🏆 TRAINING COMPLETE")
    print("="*70)
    
    if global_best_design:
        best_params, best_metrics = global_best_design
        print(f"\n✨ Best Design Found:")
        print(f"   PAR:                  {best_params['PAR']}")
        print(f"   BUFFER_DEPTH:         {best_params.get('BUFFER_DEPTH', 1024)}")
        print(f"\n📊 Best Metrics:")
        print(f"   Total Cells:          {best_metrics.get('total_cells', 'N/A')}")
        print(f"   Flip-Flops:           {best_metrics.get('flip_flops', 'N/A')}")
        print(f"   Logic Cells:          {best_metrics.get('logic_cells', 'N/A')}")
        print(f"   Throughput:           {best_metrics.get('throughput', 'N/A')} ops/cycle")
        print(f"   Area Efficiency:      {best_metrics.get('area_per_throughput', 'N/A'):.1f} cells/op")
        print(f"   Objective Score:      {global_best_objective:.1f}")
        
        # Check if best design meets all constraints
        if best_metrics.get('constraints_violated'):
            print(f"\n⚠️  WARNING: Best design violates constraints:")
            for violation in best_metrics['constraints_violated']:
                print(f"   • {violation}")
        else:
            print(f"\n✅ Best design meets all constraints!")
    
    print("\n" + "="*70)
    
    # Generate reports
    try:
        generate_all_reports(all_history, global_best_design)
        plot_training_curves(episode_rewards, episode_best_objectives, agent.losses)
    except Exception as e:
        print(f"\n⚠️  Report generation failed: {e}")
    
    # Generate Verilog for best design
    best_par = best_params["PAR"]
    best_buffer_depth = best_params.get("BUFFER_DEPTH", 1024)
    best_addr_width = int(math.ceil(math.log2(best_buffer_depth)))

    best_rtl = f"""
module reduce_sum #(
    parameter PAR = {best_par},
    parameter BUFFER_DEPTH = {best_buffer_depth}
) (
    input clk,
    input rst,
    input [31:0] in_data,
    input in_valid,
    output reg [31:0] out_data,
    output reg out_valid
);

reg [31:0] acc [0:PAR-1];
reg [{best_addr_width-1}:0] count;
integer i;

reg [31:0] final_sum;

always @(posedge clk) begin
    if (rst) begin
        for (i = 0; i < PAR; i = i + 1)
            acc[i] <= 0;
        count <= 0;
        out_valid <= 0;
    end
    else if (in_valid) begin
        for (i = 0; i < PAR; i = i + 1)
            acc[i] <= acc[i] + in_data + i;

        count <= count + 1;

        if (count == BUFFER_DEPTH - 1) begin
            final_sum = 0;
            for (i = 0; i < PAR; i = i + 1)
                final_sum = final_sum + acc[i];

            out_data <= final_sum;
            out_valid <= 1;
            count <= 0;
        end
    end
end

endmodule
"""
    with open("../../rtl/best_design.v", "w") as f:
        f.write(best_rtl)
    print(f"\n[SUCCESS] Best design Verilog code saved to rtl/best_design.v")

    return agent, all_history, global_best_design


def run_evaluation(agent, iterations=20):
    """
    Evaluate trained DQN agent (no exploration, greedy actions only).
    
    Args:
        agent: Trained DQNAgent
        iterations: Number of designs to evaluate
    """
    
    print("\n" + "="*70)
    print(" 🎯 DQN EVALUATION MODE - EXPLOITING LEARNED POLICY")
    print("="*70)
    print(f"\n📋 Evaluation Configuration:")
    print(f"   Iterations: {iterations}")
    print(f"   Mode: Greedy (no exploration)")
    print("\n" + "="*70)
    
    history = []
    best_design = None
    best_objective = float('inf')
    
    for i in range(iterations):
        # Get current state
        state = agent.encode_state(history)
        
        # Agent selects action (greedy, no exploration)
        action_idx, params = agent.select_action(state, evaluation=True)
        
        # Evaluate design
        debug_mode = (i == 0)
        objective, metrics = evaluate_design(params, i, debug=debug_mode)
        
        # Track best design
        if objective < best_objective:
            best_objective = objective
            best_design = (params.copy(), metrics.copy())
        
        history.append((params, metrics))
        
        # Display iteration results
        par = params['PAR']
        bd = params['BUFFER_DEPTH']
        print(f"\n{'='*70}")
        print(f"Iteration {i+1}/{iterations}: PAR={par}, BUFFER_DEPTH={bd}")
        print(f"{'='*70}")
        print(f"  📊 Hardware Metrics:")
        print(f"     Total Cells:        {metrics.get('total_cells', 'N/A'):>6}")
        print(f"     Flip-Flops:         {metrics.get('flip_flops', 'N/A'):>6}")
        print(f"     Logic Cells:        {metrics.get('logic_cells', 'N/A'):>6}")
        print(f"  🎯 Performance Metrics:")
        print(f"     Throughput:         {par:>6} ops/cycle")
        print(f"     Area/Throughput:    {metrics.get('area_per_throughput', 'N/A'):>6.1f} cells/op")
        print(f"  📈 Optimization:")
        print(f"     Objective (AEP):    {objective:>6.1f}")
        print(f"     Best So Far:        {best_objective:>6.1f}")
        
        # Display constraint violations if any
        if metrics.get('constraints_violated'):
            print(f"  ⚠️  Constraint Violations:")
            for violation in metrics['constraints_violated']:
                print(f"     • {violation}")
    
    # Final summary
    print("\n\n" + "="*70)
    print(" 🏆 EVALUATION COMPLETE")
    print("="*70)
    
    if best_design:
        best_params, best_metrics = best_design
        print(f"\n✨ Best Design Found:")
        print(f"   PAR:                  {best_params['PAR']}")
        print(f"   BUFFER_DEPTH:         {best_params.get('BUFFER_DEPTH', 1024)}")
        print(f"\n📊 Best Metrics:")
        print(f"   Total Cells:          {best_metrics.get('total_cells', 'N/A')}")
        print(f"   Flip-Flops:           {best_metrics.get('flip_flops', 'N/A')}")
        print(f"   Logic Cells:          {best_metrics.get('logic_cells', 'N/A')}")
        print(f"   Throughput:           {best_metrics.get('throughput', 'N/A')} ops/cycle")
        print(f"   Area Efficiency:      {best_metrics.get('area_per_throughput', 'N/A'):.1f} cells/op")
        print(f"   Objective Score:      {best_objective:.1f}")
        
        if best_metrics.get('constraints_violated'):
            print(f"\n⚠️  WARNING: Best design violates constraints:")
            for violation in best_metrics['constraints_violated']:
                print(f"   • {violation}")
        else:
            print(f"\n✅ Best design meets all constraints!")
    
    print("\n" + "="*70)
    
    try:
        generate_all_reports(history, best_design)
    except Exception as e:
        print(f"\n⚠️  Report generation failed: {e}")
    
    # Generate Verilog for best design
    best_par = best_params["PAR"]
    best_buffer_depth = best_params.get("BUFFER_DEPTH", 1024)
    best_addr_width = int(math.ceil(math.log2(best_buffer_depth)))

    best_rtl = f"""
module reduce_sum #(
    parameter PAR = {best_par},
    parameter BUFFER_DEPTH = {best_buffer_depth}
) (
    input clk,
    input rst,
    input [31:0] in_data,
    input in_valid,
    output reg [31:0] out_data,
    output reg out_valid
);

reg [31:0] acc [0:PAR-1];
reg [{best_addr_width-1}:0] count;
integer i;

reg [31:0] final_sum;

always @(posedge clk) begin
    if (rst) begin
        for (i = 0; i < PAR; i = i + 1)
            acc[i] <= 0;
        count <= 0;
        out_valid <= 0;
    end
    else if (in_valid) begin
        for (i = 0; i < PAR; i = i + 1)
            acc[i] <= acc[i] + in_data + i;

        count <= count + 1;

        if (count == BUFFER_DEPTH - 1) begin
            final_sum = 0;
            for (i = 0; i < PAR; i = i + 1)
                final_sum = final_sum + acc[i];

            out_data <= final_sum;
            out_valid <= 1;
            count <= 0;
        end
    end
end

endmodule
"""
    with open("../../rtl/best_design.v", "w") as f:
        f.write(best_rtl)
    print(f"\n[SUCCESS] Best design Verilog code saved to rtl/best_design.v")

    return history, best_design


def plot_training_curves(episode_rewards, episode_objectives, losses):
    """Plot training progress"""
    try:
        os.makedirs('../../results/reinforcement_learning', exist_ok=True)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Episode rewards
        axes[0].plot(episode_rewards, 'b-', alpha=0.3)
        axes[0].plot(np.convolve(episode_rewards, np.ones(5)/5, mode='valid'), 'b-', linewidth=2)
        axes[0].set_xlabel('Episode')
        axes[0].set_ylabel('Total Reward')
        axes[0].set_title('Episode Rewards (5-episode moving avg)')
        axes[0].grid(True, alpha=0.3)
        
        # Best objectives per episode
        axes[1].plot(episode_objectives, 'r-', alpha=0.6, linewidth=2)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Best Objective')
        axes[1].set_title('Best Design Objective per Episode')
        axes[1].grid(True, alpha=0.3)
        
        # Training loss
        if losses:
            axes[2].plot(losses, 'g-', alpha=0.3)
            axes[2].plot(np.convolve(losses, np.ones(50)/50, mode='valid'), 'g-', linewidth=2)
            axes[2].set_xlabel('Training Step')
            axes[2].set_ylabel('Loss')
            axes[2].set_title('Training Loss (50-step moving avg)')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('results/reinforcement_learning/training_curves.png', dpi=150)
        print(f"\n📈 Training curves saved to results/reinforcement_learning/training_curves.png")
        plt.close()
    except Exception as e:
        print(f"⚠️  Could not plot training curves: {e}")


def main():
    parser = argparse.ArgumentParser(description='DQN Hardware Design Optimization')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'evaluate'],
                        help='Mode: train or evaluate')
    parser.add_argument('--episodes', type=int, default=50,
                        help='Number of training episodes')
    parser.add_argument('--iterations', type=int, default=20,
                        help='Iterations per episode (train) or total iterations (evaluate)')
    parser.add_argument('--load', type=str, default=None,
                        help='Load checkpoint file (e.g., reinforcement_learning/checkpoints/dqn_final.pt)')
    parser.add_argument('--save', type=str, default='reinforcement_learning/checkpoints/dqn_final.pt',
                        help='Save checkpoint file')
    
    args = parser.parse_args()
    
    # Create directories
    os.makedirs('../checkpoints', exist_ok=True)
    os.makedirs('../../results/reinforcement_learning', exist_ok=True)
    
    # Initialize agent
    agent = DQNAgent(
        state_dim=16,
        lr=0.001,
        gamma=0.95,
        epsilon_start=1.0 if args.mode == 'train' else 0.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        batch_size=32,
        target_update_freq=10
    )
    
    # Load checkpoint if specified
    if args.load:
        agent.load(args.load)
    
    # Run training or evaluation
    if args.mode == 'train':
        agent, history, best_design = run_training(
            episodes=args.episodes,
            iterations_per_episode=args.iterations,
            agent=agent
        )
        # Agent already saved during training
    else:
        history, best_design = run_evaluation(agent, iterations=args.iterations)


if __name__ == '__main__':
    main()
