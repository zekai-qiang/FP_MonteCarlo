import pandas as pd
import numpy as np
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
from multiprocessing import Pool, cpu_count
import time
import numba
from numba import jit, prange

warnings.filterwarnings('ignore')

@dataclass
class FoundationSchool:
    name: str
    programmes: int
    first_choice_preference: int
    ratio_after_preallocation: float
    year: str = None
    is_forecasted: bool = False

@jit(nopython=True, parallel=True)
def generate_all_preferences_batch(first_choices, school_ratios, similarity_weights_cache, 
                                  n_schools, n_applicants, preference_correlation):
    """Uses Numba JIT"""
    all_preferences = np.zeros((n_applicants, n_schools), dtype=np.int32)
    
    for i in prange(n_applicants):
        first_choice_idx = first_choices[i]
        all_preferences[i, 0] = first_choice_idx
        
        # Get pre-computed similarity weights
        sim_weights = similarity_weights_cache[first_choice_idx].copy()
        sim_weights[first_choice_idx] = 0.0  # Zero out first choice
        
        # Add random factors
        random_factors = np.random.uniform(0.5, 1.5, n_schools)
        weights = sim_weights * preference_correlation + random_factors * (1 - preference_correlation)
        weights[first_choice_idx] = 0.0
        
        # Normalize non-zero weights
        total_weight = np.sum(weights)
        if total_weight > 0:
            weights = weights / total_weight
        
        # Generate remaining preferences using weighted sampling
        remaining_schools = np.arange(n_schools)
        for pos in range(1, n_schools):
            if np.sum(weights) == 0:
                break
            
            # Sample next school
            cumsum = np.cumsum(weights)
            rand_val = np.random.random() * cumsum[-1]
            chosen_idx = np.searchsorted(cumsum, rand_val)
            
            if chosen_idx >= n_schools:
                chosen_idx = n_schools - 1
            
            # Find actual school index
            selected_school = remaining_schools[chosen_idx]
            all_preferences[i, pos] = selected_school
            
            # Remove selected school from consideration
            weights = np.concatenate((weights[:chosen_idx], weights[chosen_idx+1:]))
            remaining_schools = np.concatenate((remaining_schools[:chosen_idx], 
                                              remaining_schools[chosen_idx+1:]))
    
    return all_preferences

@jit(nopython=True)
def run_pia_jit(preferences, school_programmes, n_applicants, n_schools):
    """JIT-compiled PIA allocation for maximum speed"""
    remaining_capacity = school_programmes.copy()
    allocations = np.full(n_applicants, -1, dtype=np.int32)
    
    # Generate random applicant order
    applicant_order = np.arange(n_applicants)
    np.random.shuffle(applicant_order)
    
    # PASS 1: First choice only
    for i in range(n_applicants):
        applicant_id = applicant_order[i]
        first_choice_idx = preferences[applicant_id, 0]
        if remaining_capacity[first_choice_idx] > 0:
            allocations[applicant_id] = first_choice_idx
            remaining_capacity[first_choice_idx] -= 1
    
    # PASS 2: Remaining preferences
    for i in range(n_applicants):
        applicant_id = applicant_order[i]
        if allocations[applicant_id] >= 0:  # Already allocated
            continue
            
        for pos in range(n_schools):
            school_idx = preferences[applicant_id, pos]
            if remaining_capacity[school_idx] > 0:
                allocations[applicant_id] = school_idx
                remaining_capacity[school_idx] -= 1
                break
    
    return allocations

class UKFPOSimulator:
    def __init__(self, year_data: Dict[str, FoundationSchool]):
        self.schools = year_data
        self.school_names = list(year_data.keys())
        self.n_schools = len(self.school_names)
        self._precompute_simulation_data()
    
    def _precompute_simulation_data(self):
        """Pre-compute expensive calculations"""
        # Basic mappings
        self.school_to_idx = {name: i for i, name in enumerate(self.school_names)}
        self.idx_to_school = {i: name for i, name in enumerate(self.school_names)}
        
        # Pre-compute arrays for JIT functions
        total_first_choices = sum(school.first_choice_preference for school in self.schools.values())
        self.first_choice_probs = np.array([
            self.schools[name].first_choice_preference / total_first_choices 
            for name in self.school_names
        ])
        
        self.school_ratios = np.array([
            self.schools[name].ratio_after_preallocation 
            for name in self.school_names
        ])
        
        self.school_programmes = np.array([
            self.schools[name].programmes 
            for name in self.school_names
        ], dtype=np.int32)
        
        self.total_applicants = sum(school.first_choice_preference for school in self.schools.values())
        
        # Pre-compute similarity weights for all schools
        self.similarity_weights_cache = np.zeros((self.n_schools, self.n_schools))
        for i in range(self.n_schools):
            ratio_diffs = np.abs(self.school_ratios - self.school_ratios[i])
            self.similarity_weights_cache[i] = 1.0 / (1.0 + ratio_diffs)
    
    def _run_batch_simulations(self, args):
        """Heavily optimized batch simulation using JIT compilation"""
        batch_size, user_preferences_idx, preference_correlation, process_id = args
        
        # Set random seeds
        np.random.seed(process_id + int(time.time() * 1000) % 100000)
        random.seed(process_id + int(time.time() * 1000) % 100000)
        
        # Convert user preferences to indices
        user_pref_indices = np.array([self.school_to_idx[school] for school in user_preferences_idx 
                                     if school in self.school_to_idx], dtype=np.int32)
        
        local_allocations = np.zeros(self.n_schools, dtype=int)
        local_positions = defaultdict(int)
        
        for _ in range(batch_size):
            # Generate first choices for all applicants at once
            first_choices = np.random.choice(self.n_schools, size=self.total_applicants, 
                                           p=self.first_choice_probs).astype(np.int32)
            
            # Generate all preference lists using JIT-compiled function
            all_preferences = generate_all_preferences_batch(
                first_choices, self.school_ratios, self.similarity_weights_cache,
                self.n_schools, self.total_applicants, preference_correlation
            )
            
            # Replace random applicant with user preferences
            user_applicant_id = random.randint(0, self.total_applicants - 1)
            all_preferences[user_applicant_id, :len(user_pref_indices)] = user_pref_indices
            
            # Run JIT-compiled PIA allocation
            allocations = run_pia_jit(
                all_preferences, self.school_programmes, self.total_applicants, self.n_schools
            )
            
            # Record user's allocation
            user_allocation = allocations[user_applicant_id]
            if user_allocation >= 0:
                local_allocations[user_allocation] += 1
                
                # Find position in user's preference list
                allocated_school_name = self.idx_to_school[user_allocation]
                try:
                    position = user_preferences_idx.index(allocated_school_name) + 1
                    local_positions[position] += 1
                except ValueError:
                    local_positions['unranked'] += 1
            else:
                local_positions['unallocated'] += 1
        
        return local_allocations, dict(local_positions)
    
    def simulate_user_allocation(self, user_preferences: List[str], 
                                        n_simulations: int = 15000, 
                                        n_processes: int = None,
                                        preference_correlation: float = 0.3) -> Dict:
        """Optimized parallel simulation with JIT compilation"""
        if n_processes is None:
            n_processes = min(cpu_count(), max(1, n_simulations // 1000))
        
        print(f"Running {n_simulations} simulations using {n_processes} parallel processes...")
        start_time = time.time()
        
        # Pre-compile JIT functions with a small test run 
        test_first_choices = np.array([0, 1, 2], dtype=np.int32)
        test_prefs = generate_all_preferences_batch(
            test_first_choices, self.school_ratios[:3], self.similarity_weights_cache[:3, :3],
            3, 3, preference_correlation
        )
        test_allocs = run_pia_jit(test_prefs, np.array([10, 10, 10], dtype=np.int32), 3, 3)
        
        # Calculate batch sizes for better load balancing
        base_batch_size = max(50, n_simulations // n_processes)
        remainder = n_simulations % n_processes
        
        process_args = []
        for process_id in range(n_processes):
            batch_size = base_batch_size + (1 if process_id < remainder else 0)
            if batch_size > 0:
                process_args.append((batch_size, user_preferences, preference_correlation, process_id))
        
        # Run simulations in parallel
        with Pool(processes=n_processes) as pool:
            results = pool.map(self._run_batch_simulations, process_args)
        
        # Aggregate results
        total_allocations = np.zeros(self.n_schools, dtype=int)
        total_positions = defaultdict(int)
        
        for allocation_counts, position_counts in results:
            total_allocations += allocation_counts
            for pos, count in position_counts.items():
                total_positions[pos] += count
        
        elapsed_total = time.time() - start_time
        print(f" Completed {n_simulations} simulations in {elapsed_total:.1f} seconds")
        print(f" Speed: {n_simulations/elapsed_total:.0f} simulations/second")
        
        # Convert to probabilities
        allocation_probabilities = {
            self.idx_to_school[i]: count / n_simulations 
            for i, count in enumerate(total_allocations) if count > 0
        }
        
        position_probabilities = {
            pos: count / n_simulations 
            for pos, count in total_positions.items()
        }
        
        return {
            'allocation_probabilities': allocation_probabilities,
            'position_probabilities': position_probabilities,
            'total_simulations': n_simulations,
            'execution_time': elapsed_total,
            'processes_used': n_processes
        }

def load_ukfp(excel_file: str, sheet_name: str) -> Dict[str, FoundationSchool]:
    """Load UKFP data from Excel file"""
    try:
        df = pd.read_excel(excel_file, sheet_name=sheet_name)
        
        required_columns = ['Foundation school', 'Number of programmes', 
                           'First choice preference', 
                           'First preference ratio after pre-allocation / manual matching']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        ukfp_data = {}
        for _, row in df.iterrows():
            school_name = row['Foundation school']
            programmes = int(row['Number of programmes'])
            first_choice_preference = int(row['First choice preference'])
            ratio_after_preallocation = float(row['First preference ratio after pre-allocation / manual matching'])
            
            ukfp_data[school_name] = FoundationSchool(
                school_name, programmes, first_choice_preference,
                ratio_after_preallocation, year=sheet_name
            )
        
        return ukfp_data
        
    except Exception as e:
        print(f"Error loading data from {excel_file}: {str(e)}")
        return {}

def forecast_2026_data(data_2024: Dict[str, FoundationSchool], 
                      data_2025: Dict[str, FoundationSchool]) -> Dict[str, FoundationSchool]:
    """Generate 2026 forecast using linear interpolation"""
    programmes_2026 = {
        'East of England': 809, 'London': 1175, 'KSS': 647, 'Northern': 493,
        'LNR': 270, 'North West of England': 1046, 'Northern Ireland': 363,
        'Peninsula': 302, 'Scotland': 1101, 'Severn': 397, 'Thames Valley Oxford': 336,
        'Trent': 455, 'Wales': 440, 'Wessex': 407, 'West Midlands Central': 278,
        'West Midlands North': 416, 'West Midlands South': 284, 'Yorkshire and Humber': 824
    }
    
    forecasted_2026 = {}
    for school_name in data_2025.keys():
        if school_name not in programmes_2026:
            continue
        
        school_2024 = data_2024[school_name]
        school_2025 = data_2025[school_name]
        programmes_2026_count = programmes_2026[school_name]
        
        ratio_2024 = school_2024.ratio_after_preallocation
        ratio_2025 = school_2025.ratio_after_preallocation
        predicted_ratio_2026 = ratio_2025 + (ratio_2025 - ratio_2024)
        
        predicted_first_choices = max(50, min(5000, int(round(predicted_ratio_2026 * programmes_2026_count))))
        final_predicted_ratio = predicted_first_choices / programmes_2026_count
        
        forecasted_2026[school_name] = FoundationSchool(
            name=school_name, programmes=programmes_2026_count,
            first_choice_preference=predicted_first_choices,
            ratio_after_preallocation=final_predicted_ratio,
            year='2026', is_forecasted=True
        )
        
    return forecasted_2026

def load_all_data(excel_file_path: str):
    """Load all data and return simulators"""
    data_2024 = load_ukfp(excel_file_path, '2024')
    data_2025 = load_ukfp(excel_file_path, '2025')
    
    if not data_2024 or not data_2025:
        return None, None, None, None
    
    data_2026 = forecast_2026_data(data_2024, data_2025)
    
    simulator_2025 = UKFPOSimulator(data_2025)
    simulator_2026 = UKFPOSimulator(data_2026)
    
    return data_2025, data_2026, simulator_2025, simulator_2026

def analyze_preferences(simulator: UKFPOSimulator, user_preferences: list[str], 
                       n_simulations: int = 15000, n_processes: int = None) -> dict:
    """Analyze input preference"""
    
    # Store original user preferences for return value
    original_preferences = user_preferences.copy()
    
    # Validate user preferences
    valid_schools = set(simulator.school_names)
    invalid_schools = [school for school in user_preferences if school not in valid_schools] 

    if invalid_schools:
        error_message = f"ERROR Invalid school: {invalid_schools}"
        print(error_message)
        print("Valid school names are:")
        for i, school in enumerate(sorted(simulator.school_names), 1):
            print(f"  {i:2d}. {school}")
        
        return {
            'error': error_message,
            'invalid_schools': invalid_schools,
            'valid_schools': sorted(simulator.school_names),
            'original_preferences': original_preferences,
            'status': 'ERROR'
        }

    # If all schools are valid, proceed with simulation
    # Run parallel simulation
    results = simulator.simulate_user_allocation(
        user_preferences, n_simulations, n_processes)
    
    print(f"\n ALLOCATION PROBABILITIES:") 
    print("=" * 60)
    
    # Display preferences in original order with their probabilities
    for i, school in enumerate(original_preferences, 1):
        allocation_prob = results['allocation_probabilities'].get(school, 0.0)
        print(f" {i:2d}. {school:<25} {allocation_prob:>6.1%}")
    
    print(f"\n FATE (Top 5):") 
    print("=" * 60)
    
    # Show top probabilities for reference
    sorted_probs = sorted(results['allocation_probabilities'].items(), 
                         key=lambda x: x[1], reverse=True)
    
    for i, (school, prob) in enumerate(sorted_probs[:5], 1):  # Show top 5 only
        try:
            position = original_preferences.index(school) + 1
            position_str = f"(#{position} choice)"
        except ValueError:
            position_str = "(unranked)"
        
        print(f" {i:2d}. {school:<25} {prob:>6.1%} {position_str}")  
    
    # Performance stats
    print(f"\n PERFORMANCE:") 
    print("=" * 60)
    print(f" Execution time: {results['execution_time']:.1f} s")
    print(f" Simulations per second: {n_simulations/results['execution_time']:.0f}")
    print(f" Parallel processes used: {results['processes_used']}")
    
    # Create preference list with probabilities for each school
    preference_list_with_probabilities = []
    
    for i, school in enumerate(original_preferences, 1):
        # Get allocation probability for this school (0.0 if not in results)
        allocation_prob = results['allocation_probabilities'].get(school, 0.0)
        
        preference_list_with_probabilities.append({
            'rank': i,
            'school': school,
            'allocation_probability': allocation_prob,
            'is_valid_school': True  # All schools are valid at this point
        })
    
    # Create return dictionary
    return_data = {
        'original_preferences': original_preferences,
        'preference_list_with_probabilities': preference_list_with_probabilities,
        'simulation_results': results,
        'invalid_schools': [],  # Empty since we enforce validity
        'valid_preferences_used': user_preferences,
        'status': 'SUCCESS'
    }
    
    return return_data

