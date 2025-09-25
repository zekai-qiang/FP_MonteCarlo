from fp_monte_carlo import UKFPOSimulator, load_all_data, analyze_preferences 

def main():
    """Run Monte Carlo Simulation"""
    
    # Configuration
    excel_file_path = 'path/to/FP_ratios.xlsx'  # Update this path
    n_simulations = 15000
    n_processes = None  # Auto-detect
    
    # Input preference list
    input_preferences = [
        'London', 'Northern', 'East of England', 'Yorkshire and Humber',
        'Trent', 'Scotland', 'Wessex', 'LNR', 'Peninsula', 'Severn',
        'West Midlands North', 'Northern Ireland', 'West Midlands South',
        'West Midlands Central', 'KSS', 'North West of England',
        'Thames Valley Oxford', 'Wales'
    ] 
    
    data_2025, data_2026, simulator_2025, simulator_2026 = load_all_data(excel_file_path) 
    
    analyze_preferences(simulator_2025, input_preferences, n_simulations, n_processes)

if __name__ == "__main__":
    main()


