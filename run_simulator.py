from fp_monte_carlo import UKFPOSimulator, load_all_data, analyze_preferences 

def main():
    """Run Monte Carlo Simulation""" 
    
    excel_file_path = 'path/to/FP_ratios.xlsx'  # Update this path
    n_simulations = 15000
    n_processes = None  # Auto-detect
    
    # Input preference list

    input_preferences = [
    # 'Peninsula', 'Wales', 'Northern Ireland', 'KSS', 
    # 'Thames Valley Oxford', 'Scotland', 'East of England', 'LNR',
    # 'West Midlands Central', 'Yorkshire and Humber', 'Wessex', 'Severn',
    # 'West Midlands North', 'London', 'Northern', 'North West of England',
    # 'Trent', 'West Midlands South'
    ] 
    
    data_2025, data_2026, simulator_2025, simulator_2026 = load_all_data(excel_file_path) 
    
    analyze_preferences(simulator_2025, input_preferences, n_simulations, n_processes)

if __name__ == "__main__":
    main()


