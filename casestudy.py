"""
Acme Optimization Platform Case Study
Copyright © 2025 Mohith Nagendra. All rights reserved.

This code/program is the sole and exclusive intellectual property of Mohith Nagendra.
Unauthorized copying, modification, distribution, or use of this code, in whole or in part,
is strictly prohibited without the express written consent of Mohith Nagendra.

Author: Mohith Nagendra
Date: 2025-02-24
"""

import pandas as pd
import numpy as np
from scipy.optimize import linprog, minimize
import pulp as pl
import json
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt

# Seed for reproducibility
np.random.seed(42)

class AcmeOptimizer:
    """
    A class to optimize sales and margins for Acme based on hierarchical structure
    and various constraints at different levels.
    """
    
    def __init__(self):
        """Initialize the optimizer with empty structures"""
        # Define the hierarchy levels
        self.levels = ['Portfolio', 'Geography', 'Category', 'Brand', 'Segment']
        
        # Store the data in a structured format
        self.data = {}
        
        # Store the constraints
        self.constraints = {}
        
        # Store the optimized results
        self.results = {}
        
    def create_synthetic_data(self):
        """
        Create a synthetic data set that represents Acme's structure
        """
        # Define the structure based on the information provided
        structure = {
            "Acme": {
                "North America": {
                    "Fragrance": {
                        "Kilian": {
                            "Perfume": {"sales": 5000000, "margin": 0.45}
                        },
                        "Balmain": {
                            "Cologne": {"sales": 4000000, "margin": 0.35}
                        },
                        "Frederic Malle": {
                            "Essential Oils": {"sales": 3000000, "margin": 0.40}
                        }
                    },
                    "Color Cosmetics": {
                        "Bobbi Brown": {
                            "Lipstick": {"sales": 2500000, "margin": 0.40},
                            "Mascara": {"sales": 2000000, "margin": 0.15}
                        },
                        "Elizabeth Arden": {
                            "Foundation": {"sales": 3500000, "margin": 0.25},
                            "Mascara": {"sales": 1400000, "margin": 0.20},
                            "Bronzer": {"sales": 1200000, "margin": 0.30}
                        }
                    }
                },
                "South America": {
                    "Hair": {
                        "Aveda": {
                            "Shampoo": {"sales": 2000000, "margin": 0.30},
                            "Conditioner": {"sales": 1800000, "margin": 0.35}
                        }
                    },
                    "Body": {
                        "Origins": {
                            "Lotion": {"sales": 1500000, "margin": 0.40},
                            "Scrub": {"sales": 1200000, "margin": 0.45}
                        }
                    }
                },
                "Asia": {
                    "Fragrance": {
                        "Balmain": {
                            "Cologne": {"sales": 2500000, "margin": 0.30}
                        }
                    },
                    "Face Make-up": {
                        "Bobbi Brown": {
                            "Foundation": {"sales": 3000000, "margin": 0.35}
                        },
                        "Elizabeth Arden": {
                            "Mascara": {"sales": 1800000, "margin": 0.20},
                            "Bronzer": {"sales": 1500000, "margin": 0.30}
                        }
                    }
                },
                "Europe": {
                    "Hair Dye": {
                        "Brand1": {
                            "Product1": {"sales": 4000000, "margin": 0.25}
                        }
                    },
                    "Make-up brushes": {
                        "Brand2": {
                            "Product2": {"sales": 3000000, "margin": 0.30}
                        }
                    },
                    "Face make-up": {
                        "Brand3": {
                            "Product3": {"sales": 2000000, "margin": 0.35}
                        }
                    },
                    "Fragrance": {
                        "Kilian": {
                            "Perfume": {"sales": 3000000, "margin": 0.40}
                        },
                        "Balmain": {
                            "Cologne": {"sales": 2500000, "margin": 0.35}
                        },
                        "Frederic Malle": {
                            "Essential Oils": {"sales": 2000000, "margin": 0.40}
                        }
                    },
                    "Tools": {
                        "Bobbi Brown": {
                            "Lipstick": {"sales": 3400000, "margin": 0.45},
                            "Mascara": {"sales": 3000000, "margin": 0.10}
                        },
                        "Elizabeth Arden": {
                            "Lipstick": {"sales": 11000000, "margin": 0.23},
                            "Toner": {"sales": 3000000, "margin": 0.08},
                            "Bronzer": {"sales": 4000000, "margin": 0.30},
                            "Mascara": {"sales": 4000000, "margin": 0.17}
                        }
                    }
                }
            }
        }
        
        # Convert the nested dictionary to a flat DataFrame for easier manipulation
        rows = []
        
        def traverse_dict(d, path=None):
            if path is None:
                path = []
                
            for k, v in d.items():
                new_path = path + [k]
                
                if isinstance(v, dict) and "sales" in v and "margin" in v:
                    row = {
                        "Portfolio": path[0] if len(path) > 0 else None,
                        "Geography": path[1] if len(path) > 1 else None,
                        "Category": path[2] if len(path) > 2 else None,
                        "Brand": path[3] if len(path) > 3 else None,
                        "Segment": k,
                        "Sales": v["sales"],
                        "Margin": v["margin"],
                        "Profit": v["sales"] * v["margin"]
                    }
                    rows.append(row)
                elif isinstance(v, dict):
                    traverse_dict(v, new_path)
        
        traverse_dict(structure)
        self.df = pd.DataFrame(rows)
        
        # Calculate contribution at each level
        for level in self.levels:
            if level == 'Portfolio':
                # Calculate total sales
                total_sales = self.df['Sales'].sum()
                self.df[f'{level}_Contribution'] = self.df.groupby(level)['Sales'].transform('sum') / total_sales
            else:
                # Get the parent level
                parent_level_idx = self.levels.index(level) - 1
                parent_level = self.levels[parent_level_idx]
                
                # Calculate contribution within parent level
                self.df[f'{level}_Contribution'] = self.df.groupby([parent_level, level])['Sales'].transform('sum') / self.df.groupby(parent_level)['Sales'].transform('sum')
        
        # Add initial trend (for demonstration purposes)
        self.df['Trend'] = np.random.uniform(-0.05, 0.15, size=len(self.df))
        
        # Return the dataframe
        return self.df
    
    def apply_constraints(self, df, constraints):
        """
        Apply the constraints to the data
        
        Args:
            df: DataFrame with the data
            constraints: Dictionary with constraints
            
        Returns:
            DataFrame with constraints applied
        """
        # Apply global constraints
        for level in self.levels:
            for unit in df[level].unique():
                # Check if there are constraints for this unit
                unit_key = f"{level}:{unit}"
                if unit_key in constraints:
                    unit_constraints = constraints[unit_key]
                    
                    # Apply min/max trend
                    if 'min_trend' in unit_constraints:
                        df.loc[df[level] == unit, 'Trend'] = np.maximum(df.loc[df[level] == unit, 'Trend'], unit_constraints['min_trend'])
                    if 'max_trend' in unit_constraints:
                        df.loc[df[level] == unit, 'Trend'] = np.minimum(df.loc[df[level] == unit, 'Trend'], unit_constraints['max_trend'])
                    
                    # Apply min/max contribution (this is more complex and would need optimization)
                    # For simplicity, we'll just note it here
        
        # Apply branch constraints (this would need a more complex implementation)
        # For simplicity, we'll just return the dataframe with global constraints applied
        return df
    
    def optimize_sales(self, constraints, years=1):
        """
        Optimize to maximize sales given constraints
        
        Args:
            constraints: Dictionary with constraints
            years: Number of years to project (default is 1)
            
        Returns:
            Optimized sales for each segment
        """
        # Create a copy of the data to work with
        df = self.df.copy()
        
        # Use PuLP for optimization
        model = pl.LpProblem("Maximize_Sales", pl.LpMaximize)
        
        # Define the decision variables (trend adjustments)
        trend_vars = {}
        for idx, row in df.iterrows():
            var_name = f"trend_{idx}"
            # Define the bounds based on constraints
            segment = row['Segment']
            brand = row['Brand']
            category = row['Category']
            geography = row['Geography']
            
            # Default bounds
            lower_bound = -0.1  # Assume -10% min trend if not specified
            upper_bound = 0.2   # Assume 20% max trend if not specified
            
            # Check for specific constraints
            segment_key = f"Segment:{segment}"
            brand_key = f"Brand:{brand}"
            category_key = f"Category:{category}"
            geography_key = f"Geography:{geography}"
            
            if segment_key in constraints and 'min_trend' in constraints[segment_key]:
                lower_bound = constraints[segment_key]['min_trend']
            if segment_key in constraints and 'max_trend' in constraints[segment_key]:
                upper_bound = constraints[segment_key]['max_trend']
                
            # Create the variable
            trend_vars[idx] = pl.LpVariable(var_name, lower_bound, upper_bound)
        
        # Define the objective function (maximize total sales)
        model += pl.lpSum([df.loc[idx, 'Sales'] * (1 + trend_vars[idx]) for idx in df.index])
        
        # Add constraints for contribution
        # This would be more complex in a real implementation
        # For simplicity, we'll skip this part
        
        # Solve the model
        model.solve()
        
        # Extract the results
        results = df.copy()
        for idx in df.index:
            results.loc[idx, 'New_Trend'] = trend_vars[idx].value()
            results.loc[idx, 'New_Sales'] = df.loc[idx, 'Sales'] * (1 + trend_vars[idx].value())
            results.loc[idx, 'New_Profit'] = results.loc[idx, 'New_Sales'] * df.loc[idx, 'Margin']
        
        # Calculate new contributions
        total_new_sales = results['New_Sales'].sum()
        results['New_Portfolio_Contribution'] = results.groupby('Portfolio')['New_Sales'].transform('sum') / total_new_sales
        
        # For other levels, calculate contribution within parent
        for i, level in enumerate(self.levels[1:], 1):
            parent_level = self.levels[i-1]
            results[f'New_{level}_Contribution'] = (
                results.groupby([parent_level, level])['New_Sales'].transform('sum') / 
                results.groupby(parent_level)['New_Sales'].transform('sum')
            )
        
        # Project for multiple years if requested
        if years > 1:
            yearly_results = {}
            yearly_results[1] = results
            
            for year in range(2, years + 1):
                # Use the previous year's results as the starting point
                prev_results = yearly_results[year - 1]
                cur_results = prev_results.copy()
                
                # Re-run the optimization with the new base
                model = pl.LpProblem(f"Maximize_Sales_Year_{year}", pl.LpMaximize)
                
                # Define new variables
                new_trend_vars = {}
                for idx, row in cur_results.iterrows():
                    var_name = f"trend_{idx}_year_{year}"
                    new_trend_vars[idx] = pl.LpVariable(var_name, lower_bound, upper_bound)
                
                # Define the objective function
                model += pl.lpSum([prev_results.loc[idx, 'New_Sales'] * (1 + new_trend_vars[idx]) for idx in cur_results.index])
                
                # Solve the model
                model.solve()
                
                # Extract the results
                for idx in cur_results.index:
                    cur_results.loc[idx, 'New_Trend'] = new_trend_vars[idx].value()
                    cur_results.loc[idx, 'New_Sales'] = prev_results.loc[idx, 'New_Sales'] * (1 + new_trend_vars[idx].value())
                    cur_results.loc[idx, 'New_Profit'] = cur_results.loc[idx, 'New_Sales'] * cur_results.loc[idx, 'Margin']
                
                # Calculate new contributions
                total_new_sales = cur_results['New_Sales'].sum()
                cur_results['New_Portfolio_Contribution'] = cur_results.groupby('Portfolio')['New_Sales'].transform('sum') / total_new_sales
                
                # For other levels, calculate contribution within parent
                for i, level in enumerate(self.levels[1:], 1):
                    parent_level = self.levels[i-1]
                    cur_results[f'New_{level}_Contribution'] = (
                        cur_results.groupby([parent_level, level])['New_Sales'].transform('sum') / 
                        cur_results.groupby(parent_level)['New_Sales'].transform('sum')
                    )
                
                yearly_results[year] = cur_results
            
            return yearly_results
        
        return results
    
    def optimize_margin(self, constraints, years=1):
        """
        Optimize to maximize margin given constraints
        
        Args:
            constraints: Dictionary with constraints
            years: Number of years to project (default is 1)
            
        Returns:
            Optimized margin for each segment
        """
        # Similar to optimize_sales but with a focus on maximizing margin
        # Create a copy of the data to work with
        df = self.df.copy()
        
        # Use PuLP for optimization
        model = pl.LpProblem("Maximize_Margin", pl.LpMaximize)
        
        # Define the decision variables (trend adjustments)
        trend_vars = {}
        for idx, row in df.iterrows():
            var_name = f"trend_{idx}"
            # Define the bounds based on constraints
            segment = row['Segment']
            brand = row['Brand']
            category = row['Category']
            geography = row['Geography']
            
            # Default bounds
            lower_bound = -0.1  # Assume -10% min trend if not specified
            upper_bound = 0.2   # Assume 20% max trend if not specified
            
            # Check for specific constraints
            segment_key = f"Segment:{segment}"
            brand_key = f"Brand:{brand}"
            category_key = f"Category:{category}"
            geography_key = f"Geography:{geography}"
            
            if segment_key in constraints and 'min_trend' in constraints[segment_key]:
                lower_bound = constraints[segment_key]['min_trend']
            if segment_key in constraints and 'max_trend' in constraints[segment_key]:
                upper_bound = constraints[segment_key]['max_trend']
                
            # Create the variable
            trend_vars[idx] = pl.LpVariable(var_name, lower_bound, upper_bound)
        
        # Define the objective function (maximize total profit)
        model += pl.lpSum([df.loc[idx, 'Sales'] * (1 + trend_vars[idx]) * df.loc[idx, 'Margin'] for idx in df.index])
        
        # Solve the model
        model.solve()
        
        # Extract the results
        results = df.copy()
        for idx in df.index:
            results.loc[idx, 'New_Trend'] = trend_vars[idx].value()
            results.loc[idx, 'New_Sales'] = df.loc[idx, 'Sales'] * (1 + trend_vars[idx].value())
            results.loc[idx, 'New_Profit'] = results.loc[idx, 'New_Sales'] * df.loc[idx, 'Margin']
        
        # Calculate new contributions
        total_new_sales = results['New_Sales'].sum()
        results['New_Portfolio_Contribution'] = results.groupby('Portfolio')['New_Sales'].transform('sum') / total_new_sales
        
        # For other levels, calculate contribution within parent
        for i, level in enumerate(self.levels[1:], 1):
            parent_level = self.levels[i-1]
            results[f'New_{level}_Contribution'] = (
                results.groupby([parent_level, level])['New_Sales'].transform('sum') / 
                results.groupby(parent_level)['New_Sales'].transform('sum')
            )
        
        # Project for multiple years if requested
        if years > 1:
            yearly_results = {}
            yearly_results[1] = results
            
            for year in range(2, years + 1):
                # Use the previous year's results as the starting point
                prev_results = yearly_results[year - 1]
                cur_results = prev_results.copy()
                
                # Re-run the optimization with the new base
                model = pl.LpProblem(f"Maximize_Margin_Year_{year}", pl.LpMaximize)
                
                # Define new variables
                new_trend_vars = {}
                for idx, row in cur_results.iterrows():
                    var_name = f"trend_{idx}_year_{year}"
                    new_trend_vars[idx] = pl.LpVariable(var_name, lower_bound, upper_bound)
                
                # Define the objective function
                model += pl.lpSum([prev_results.loc[idx, 'New_Sales'] * (1 + new_trend_vars[idx]) * cur_results.loc[idx, 'Margin'] for idx in cur_results.index])
                
                # Solve the model
                model.solve()
                
                # Extract the results
                for idx in cur_results.index:
                    cur_results.loc[idx, 'New_Trend'] = new_trend_vars[idx].value()
                    cur_results.loc[idx, 'New_Sales'] = prev_results.loc[idx, 'New_Sales'] * (1 + new_trend_vars[idx].value())
                    cur_results.loc[idx, 'New_Profit'] = cur_results.loc[idx, 'New_Sales'] * cur_results.loc[idx, 'Margin']
                
                # Calculate new contributions
                total_new_sales = cur_results['New_Sales'].sum()
                cur_results['New_Portfolio_Contribution'] = cur_results.groupby('Portfolio')['New_Sales'].transform('sum') / total_new_sales
                
                # For other levels, calculate contribution within parent
                for i, level in enumerate(self.levels[1:], 1):
                    parent_level = self.levels[i-1]
                    cur_results[f'New_{level}_Contribution'] = (
                        cur_results.groupby([parent_level, level])['New_Sales'].transform('sum') / 
                        cur_results.groupby(parent_level)['New_Sales'].transform('sum')
                    )
                
                yearly_results[year] = cur_results
            
            return yearly_results
        
        return results
    
    def hit_sales_target_maximize_margin(self, target_sales, constraints, years=1):
        """
        Optimize to hit a sales target while maximizing margin
        
        Args:
            target_sales: Target sales amount
            constraints: Dictionary with constraints
            years: Number of years to project (default is 1)
            
        Returns:
            Optimized results for each segment
        """
        # Create a copy of the data to work with
        df = self.df.copy()
        
        # Use PuLP for optimization
        model = pl.LpProblem("Hit_Sales_Maximize_Margin", pl.LpMaximize)
        
        # Define the decision variables (trend adjustments)
        trend_vars = {}
        for idx, row in df.iterrows():
            var_name = f"trend_{idx}"
            # Define the bounds based on constraints
            segment = row['Segment']
            brand = row['Brand']
            category = row['Category']
            geography = row['Geography']
            
            # Default bounds
            lower_bound = -0.1  # Assume -10% min trend if not specified
            upper_bound = 0.2   # Assume 20% max trend if not specified
            
            # Check for specific constraints
            segment_key = f"Segment:{segment}"
            brand_key = f"Brand:{brand}"
            category_key = f"Category:{category}"
            geography_key = f"Geography:{geography}"
            
            if segment_key in constraints and 'min_trend' in constraints[segment_key]:
                lower_bound = constraints[segment_key]['min_trend']
            if segment_key in constraints and 'max_trend' in constraints[segment_key]:
                upper_bound = constraints[segment_key]['max_trend']
                
            # Create the variable
            trend_vars[idx] = pl.LpVariable(var_name, lower_bound, upper_bound)
        
        # Define the objective function (maximize total profit)
        model += pl.lpSum([df.loc[idx, 'Sales'] * (1 + trend_vars[idx]) * df.loc[idx, 'Margin'] for idx in df.index])
        
        # Add constraint for target sales
        model += pl.lpSum([df.loc[idx, 'Sales'] * (1 + trend_vars[idx]) for idx in df.index]) >= target_sales
        
        # Solve the model
        model.solve()
        
        # Extract the results
        results = df.copy()
        for idx in df.index:
            results.loc[idx, 'New_Trend'] = trend_vars[idx].value()
            results.loc[idx, 'New_Sales'] = df.loc[idx, 'Sales'] * (1 + trend_vars[idx].value())
            results.loc[idx, 'New_Profit'] = results.loc[idx, 'New_Sales'] * df.loc[idx, 'Margin']
        
        # Calculate new contributions
        total_new_sales = results['New_Sales'].sum()
        results['New_Portfolio_Contribution'] = results.groupby('Portfolio')['New_Sales'].transform('sum') / total_new_sales
        
        # For other levels, calculate contribution within parent
        for i, level in enumerate(self.levels[1:], 1):
            parent_level = self.levels[i-1]
            results[f'New_{level}_Contribution'] = (
                results.groupby([parent_level, level])['New_Sales'].transform('sum') / 
                results.groupby(parent_level)['New_Sales'].transform('sum')
            )
        
        # Project for multiple years if requested
        if years > 1:
            yearly_results = {}
            yearly_results[1] = results
            
            for year in range(2, years + 1):
                # Use the previous year's results as the starting point
                prev_results = yearly_results[year - 1]
                cur_results = prev_results.copy()
                
                # Adjust target for each year (you might have a different logic)
                yearly_target = target_sales * (1.05 ** (year - 1))  # Assuming 5% growth in target
                
                # Re-run the optimization with the new base
                model = pl.LpProblem(f"Hit_Sales_Maximize_Margin_Year_{year}", pl.LpMaximize)
                
                # Define new variables
                new_trend_vars = {}
                for idx, row in cur_results.iterrows():
                    var_name = f"trend_{idx}_year_{year}"
                    new_trend_vars[idx] = pl.LpVariable(var_name, lower_bound, upper_bound)
                
                # Define the objective function
                model += pl.lpSum([prev_results.loc[idx, 'New_Sales'] * (1 + new_trend_vars[idx]) * cur_results.loc[idx, 'Margin'] for idx in cur_results.index])
                
                # Add constraint for target sales
                model += pl.lpSum([prev_results.loc[idx, 'New_Sales'] * (1 + new_trend_vars[idx]) for idx in cur_results.index]) >= yearly_target
                
                # Solve the model
                model.solve()
                
                # Extract the results
                for idx in cur_results.index:
                    cur_results.loc[idx, 'New_Trend'] = new_trend_vars[idx].value()
                    cur_results.loc[idx, 'New_Sales'] = prev_results.loc[idx, 'New_Sales'] * (1 + new_trend_vars[idx].value())
                    cur_results.loc[idx, 'New_Profit'] = cur_results.loc[idx, 'New_Sales'] * cur_results.loc[idx, 'Margin']
                
                # Calculate new contributions
                total_new_sales = cur_results['New_Sales'].sum()
                cur_results['New_Portfolio_Contribution'] = cur_results.groupby('Portfolio')['New_Sales'].transform('sum') / total_new_sales
                
                # For other levels, calculate contribution within parent
                for i, level in enumerate(self.levels[1:], 1):
                    parent_level = self.levels[i-1]
                    cur_results[f'New_{level}_Contribution'] = (
                        cur_results.groupby([parent_level, level])['New_Sales'].transform('sum') / 
                        cur_results.groupby(parent_level)['New_Sales'].transform('sum')
                    )
                
                yearly_results[year] = cur_results
            
            return yearly_results
        
        return results
    
    def hit_margin_target_maximize_sales(self, target_margin, constraints, years=1):
        """
        Optimize to hit a margin target while maximizing sales
        
        Args:
            target_margin: Target margin amount
            constraints: Dictionary with constraints
            years: Number of years to project (default is 1)
            
        Returns:
            Optimized results for each segment
        """
        # Create a copy of the data to work with
        df = self.df.copy()
        
        # Use PuLP for optimization
        model = pl.LpProblem("Hit_Margin_Maximize_Sales", pl.LpMaximize)
        
        # Define the decision variables (trend adjustments)
        trend_vars = {}
        for idx, row in df.iterrows():
            var_name = f"trend_{idx}"
            # Define the bounds based on constraints
            segment = row['Segment']
            brand = row['Brand']
            category = row['Category']
            geography = row['Geography']
            
            # Default bounds
            lower_bound = -0.1  # Assume -10% min trend if not specified
            upper_bound = 0.2   # Assume 20% max trend if not specified
            
            # Check for specific constraints
            segment_key = f"Segment:{segment}"
            brand_key = f"Brand:{brand}"
            category_key = f"Category:{category}"
            geography_key = f"Geography:{geography}"
            
            if segment_key in constraints and 'min_trend' in constraints[segment_key]:
                lower_bound = constraints[segment_key]['min_trend']
            if segment_key in constraints and 'max_trend' in constraints[segment_key]:
                upper_bound = constraints[segment_key]['max_trend']
                
            # Create the variable
            trend_vars[idx] = pl.LpVariable(var_name, lower_bound, upper_bound)
        
        # Define the objective function (maximize total sales)
        model += pl.lpSum([df.loc[idx, 'Sales'] * (1 + trend_vars[idx]) for idx in df.index])
        
        # Add constraint for target margin
        model += pl.lpSum([df.loc[idx, 'Sales'] * (1 + trend_vars[idx]) * df.loc[idx, 'Margin'] for idx in df.index]) >= target_margin
        
        # Solve the model
        model.solve()
        
        # Extract the results
        results = df.copy()
        for idx in df.index:
            results.loc[idx, 'New_Trend'] = trend_vars[idx].value()
            results.loc[idx, 'New_Sales'] = df.loc[idx, 'Sales'] * (1 + trend_vars[idx].value())
            results.loc[idx, 'New_Profit'] = results.loc[idx, 'New_Sales'] * df.loc[idx, 'Margin']
        
        # Calculate new contributions
        total_new_sales = results['New_Sales'].sum()
        results['New_Portfolio_Contribution'] = results.groupby('Portfolio')['New_Sales'].transform('sum') / total_new_sales
        
        # For other levels, calculate contribution within parent
        for i, level in enumerate(self.levels[1:], 1):
            parent_level = self.levels[i-1]
            results[f'New_{level}_Contribution'] = (
                results.groupby([parent_level, level])['New_Sales'].transform('sum') / 
                results.groupby(parent_level)['New_Sales'].transform('sum')
            )
        
        # Project for multiple years if requested
        if years > 1:
            yearly_results = {}
            yearly_results[1] = results
            
            for year in range(2, years + 1):
                # Use the previous year's results as the starting point
                prev_results = yearly_results[year - 1]
                cur_results = prev_results.copy()
                
                # Adjust target for each year (you might have a different logic)
                yearly_target = target_margin * (1.05 ** (year - 1))  # Assuming 5% growth in target
                
                # Re-run the optimization with the new base
                model = pl.LpProblem(f"Hit_Margin_Maximize_Sales_Year_{year}", pl.LpMaximize)
                
                # Define new variables
                new_trend_vars = {}
                for idx, row in cur_results.iterrows():
                    var_name = f"trend_{idx}_year_{year}"
                    new_trend_vars[idx] = pl.LpVariable(var_name, lower_bound, upper_bound)
                
                # Define the objective function
                model += pl.lpSum([prev_results.loc[idx, 'New_Sales'] * (1 + new_trend_vars[idx]) for idx in cur_results.index])
                
                # Add constraint for target margin
                model += pl.lpSum([prev_results.loc[idx, 'New_Sales'] * (1 + new_trend_vars[idx]) * cur_results.loc[idx, 'Margin'] for idx in cur_results.index]) >= yearly_target
                
                # Solve the model
                model.solve()
                
                # Extract the results
                for idx in cur_results.index:
                    cur_results.loc[idx, 'New_Trend'] = new_trend_vars[idx].value()
                    cur_results.loc[idx, 'New_Sales'] = prev_results.loc[idx, 'New_Sales'] * (1 + new_trend_vars[idx].value())
                    cur_results.loc[idx, 'New_Profit'] = cur_results.loc[idx, 'New_Sales'] * cur_results.loc[idx, 'Margin']
                
                # Calculate new contributions
                total_new_sales = cur_results['New_Sales'].sum()
                cur_results['New_Portfolio_Contribution'] = cur_results.groupby('Portfolio')['New_Sales'].transform('sum') / total_new_sales
                
                # For other levels, calculate contribution within parent
                for i, level in enumerate(self.levels[1:], 1):
                    parent_level = self.levels[i-1]
                    cur_results[f'New_{level}_Contribution'] = (
                        cur_results.groupby([parent_level, level])['New_Sales'].transform('sum') / 
                        results.groupby(parent_level)['New_Sales'].transform('sum')
                    )
                
                yearly_results[year] = cur_results
            
            return yearly_results
        
        return results
    
    def visualize_results(self, results, metric='Sales', level='Segment'):
        """
        Visualize the optimization results with improved formatting
        """
        # Group by the specified level
        grouped = results.groupby(level)
        
        # Get the original and new values
        if metric == 'Sales':
            original = grouped['Sales'].sum()
            new = grouped['New_Sales'].sum()
            scale_factor = 1e6  # Convert to millions
            ylabel = f'{metric} (Millions $)'
        elif metric == 'Profit':
            original = grouped.apply(lambda x: (x['Sales'] * x['Margin']).sum())
            new = grouped['New_Profit'].sum()
            scale_factor = 1e6  # Convert to millions
            ylabel = f'{metric} (Millions $)'
        
        # Create a DataFrame for visualization and scale the values
        viz_df = pd.DataFrame({
            'Original': original / scale_factor,
            'Optimized': new / scale_factor
        })
        
        # Sort by the difference
        viz_df['Difference'] = viz_df['Optimized'] - viz_df['Original']
        viz_df = viz_df.sort_values('Difference', ascending=False)
        
        # Create the plot with more space for labels
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = viz_df[['Original', 'Optimized']].plot(kind='bar', ax=ax, width=0.8)
        
        # Add labels and title
        ax.set_xlabel(level)
        ax.set_ylabel(ylabel)
        ax.set_title(f'Comparison of Original vs. Optimized {metric} by {level}')
        
        # Add value labels on top of bars with improved positioning
        for i in range(len(viz_df)):
            # Calculate y-offset for labels to prevent overlap
            max_value = max(viz_df['Original'].iloc[i], viz_df['Optimized'].iloc[i])
            y_offset = max_value * 0.02  # 2% of the maximum value
            
            # Original value
            ax.text(i - 0.2, viz_df['Original'].iloc[i] + y_offset,
                    f'${viz_df["Original"].iloc[i]:.1f}M',
                    ha='center', va='bottom', rotation=0)
            # Optimized value
            ax.text(i + 0.2, viz_df['Optimized'].iloc[i] + y_offset,
                    f'${viz_df["Optimized"].iloc[i]:.1f}M',
                    ha='center', va='bottom', rotation=0)
        
        # Add a grid
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Rotate x labels and adjust layout
        plt.xticks(rotation=45, ha='right')
        
        # Add a legend
        plt.legend(title='')
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        return fig
    
    def generate_reports(self, results, years=1):
        """
        Generate summary reports from the optimization results
        
        Args:
            results: DataFrame with optimization results
            years: Number of years in the results
            
        Returns:
            Dictionary with summary reports
        """
        def convert_to_native(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            return obj

        reports = {}
        
        if years == 1:
            # Calculate profits
            original_profit = float((results['Sales'] * results['Margin']).sum())
            
            # Generate summary for a single year
            summary = {
                'Total_Original_Sales': float(results['Sales'].sum()),
                'Total_New_Sales': float(results['New_Sales'].sum()),
                'Sales_Growth': float((results['New_Sales'].sum() - results['Sales'].sum()) / results['Sales'].sum() * 100),
                'Total_Original_Profit': original_profit,
                'Total_New_Profit': float(results['New_Profit'].sum()),
                'Profit_Growth': float((results['New_Profit'].sum() - original_profit) / original_profit * 100),
                'Average_New_Trend': float(results['New_Trend'].mean() * 100),
                'Top_Growing_Segments': convert_to_native(results.nlargest(5, 'New_Trend')[['Segment', 'New_Trend']].to_dict('records')),
                'Top_Contributing_Segments': convert_to_native(results.nlargest(5, 'New_Sales')[['Segment', 'New_Sales']].to_dict('records'))
            }
            
            reports['summary'] = summary
            
            # Generate reports by level
            for level in self.levels:
                level_report = results.groupby(level).agg({
                    'Sales': 'sum',
                    'New_Sales': 'sum',
                    'Margin': 'mean',
                    'New_Profit': 'sum',
                    'New_Trend': 'mean'
                }).reset_index()
                
                # Calculate original profit
                level_report['Original_Profit'] = level_report['Sales'] * level_report['Margin']
                
                level_report['Sales_Growth'] = (level_report['New_Sales'] - level_report['Sales']) / level_report['Sales'] * 100
                level_report['Profit_Growth'] = (level_report['New_Profit'] - level_report['Original_Profit']) / level_report['Original_Profit'] * 100
                
                reports[level] = convert_to_native(level_report.to_dict('records'))
                    
        else:
            # Generate summary for multiple years
            for year in range(1, years + 1):
                year_results = results[year]
                original_profit = float((year_results['Sales'] * year_results['Margin']).sum())
                
                summary = {
                    'Year': year,
                    'Total_Sales': float(year_results['New_Sales'].sum()),
                    'Total_Profit': float(year_results['New_Profit'].sum()),
                    'Average_Trend': float(year_results['New_Trend'].mean() * 100),
                    'Top_Growing_Segments': convert_to_native(year_results.nlargest(5, 'New_Trend')[['Segment', 'New_Trend']].to_dict('records')),
                    'Top_Contributing_Segments': convert_to_native(year_results.nlargest(5, 'New_Sales')[['Segment', 'New_Sales']].to_dict('records'))
                }
                
                if year > 1:
                    prev_year_results = results[year - 1]
                    summary['Sales_Growth'] = float((year_results['New_Sales'].sum() - prev_year_results['New_Sales'].sum()) / prev_year_results['New_Sales'].sum() * 100)
                    summary['Profit_Growth'] = float((year_results['New_Profit'].sum() - prev_year_results['New_Profit'].sum()) / prev_year_results['New_Profit'].sum() * 100)
                else:
                    summary['Sales_Growth'] = float((year_results['New_Sales'].sum() - year_results['Sales'].sum()) / year_results['Sales'].sum() * 100)
                    summary['Profit_Growth'] = float((year_results['New_Profit'].sum() - original_profit) / original_profit * 100)
                
                reports[f'summary_year_{year}'] = summary
                
                # Generate reports by level for each year
                for level in self.levels:
                    level_report = year_results.groupby(level).agg({
                        'Sales': 'sum',
                        'New_Sales': 'sum',
                        'Margin': 'mean',
                        'New_Profit': 'sum',
                        'New_Trend': 'mean'
                    }).reset_index()
                    
                    # Calculate original profit
                    level_report['Original_Profit'] = level_report['Sales'] * level_report['Margin']
                    
                    level_report['Sales_Growth'] = (level_report['New_Sales'] - level_report['Sales']) / level_report['Sales'] * 100
                    level_report['Profit_Growth'] = (level_report['New_Profit'] - level_report['Original_Profit']) / level_report['Original_Profit'] * 100
                    
                    reports[f'{level}_year_{year}'] = convert_to_native(level_report.to_dict('records'))
        
        return reports

def create_synthetic_dataset_from_requirements():
    """
    Create a synthetic dataset based on the specific requirements provided.
    """
    # Create a dataset structure according to the given breakdowns
    data = {
        "Portfolio": [],
        "Geography": [],
        "Category": [],
        "Brand": [],
        "Segment": [],
        "Sales": [],
        "Margin": [],
        "Trend_Min": [],
        "Trend_Max": [],
        "Contribution_Min": [],
        "Contribution_Max": []
    }
    
    # Add the European segments that have explicit data
    # Bobbi Brown Lipstick in Europe/Tools
    data["Portfolio"].append("Acme")
    data["Geography"].append("Europe")
    data["Category"].append("Tools")
    data["Brand"].append("Bobbi Brown")
    data["Segment"].append("Lipstick")
    data["Sales"].append(3400000)
    data["Margin"].append(0.45)
    data["Trend_Min"].append(-0.04)
    data["Trend_Max"].append(0.13)
    data["Contribution_Min"].append(0.04)
    data["Contribution_Max"].append(0.15)
    
    # Bobbi Brown Mascara in Europe/Tools
    data["Portfolio"].append("Acme")
    data["Geography"].append("Europe")
    data["Category"].append("Tools")
    data["Brand"].append("Bobbi Brown")
    data["Segment"].append("Mascara")
    data["Sales"].append(3000000)
    data["Margin"].append(0.10)
    data["Trend_Min"].append(0.02)
    data["Trend_Max"].append(0.11)
    data["Contribution_Min"].append(0.015)
    data["Contribution_Max"].append(0.10)
    
    # Elizabeth Arden Lipstick in Europe/Tools
    data["Portfolio"].append("Acme")
    data["Geography"].append("Europe")
    data["Category"].append("Tools")
    data["Brand"].append("Elizabeth Arden")
    data["Segment"].append("Lipstick")
    data["Sales"].append(11000000)
    data["Margin"].append(0.23)
    data["Trend_Min"].append(0.06)
    data["Trend_Max"].append(0.22)
    data["Contribution_Min"].append(0.03)
    data["Contribution_Max"].append(0.06)
    
    # Elizabeth Arden Toner in Europe/Tools
    data["Portfolio"].append("Acme")
    data["Geography"].append("Europe")
    data["Category"].append("Tools")
    data["Brand"].append("Elizabeth Arden")
    data["Segment"].append("Toner")
    data["Sales"].append(3000000)
    data["Margin"].append(0.08)
    data["Trend_Min"].append(0.085)
    data["Trend_Max"].append(0.114)
    data["Contribution_Min"].append(0.06)
    data["Contribution_Max"].append(0.10)
    
    # Elizabeth Arden Bronzer in Europe/Tools
    data["Portfolio"].append("Acme")
    data["Geography"].append("Europe")
    data["Category"].append("Tools")
    data["Brand"].append("Elizabeth Arden")
    data["Segment"].append("Bronzer")
    data["Sales"].append(4000000)
    data["Margin"].append(0.30)
    data["Trend_Min"].append(-0.03)
    data["Trend_Max"].append(0.14)
    data["Contribution_Min"].append(0.01)
    data["Contribution_Max"].append(0.03)
    
    # Elizabeth Arden Mascara in Europe/Tools
    data["Portfolio"].append("Acme")
    data["Geography"].append("Europe")
    data["Category"].append("Tools")
    data["Brand"].append("Elizabeth Arden")
    data["Segment"].append("Mascara")
    data["Sales"].append(4000000)
    data["Margin"].append(0.17)
    data["Trend_Min"].append(0.03)
    data["Trend_Max"].append(0.16)
    data["Contribution_Min"].append(0.06)
    data["Contribution_Max"].append(0.12)
    
    # Add the Asia segments that have explicit data
    # Elizabeth Arden Mascara in Asia/Face Make-up
    data["Portfolio"].append("Acme")
    data["Geography"].append("Asia")
    data["Category"].append("Face Make-up")
    data["Brand"].append("Elizabeth Arden")
    data["Segment"].append("Mascara")
    data["Sales"].append(1400000)
    data["Margin"].append(0.20)
    data["Trend_Min"].append(-0.08)
    data["Trend_Max"].append(-0.05)
    data["Contribution_Min"].append(0.02)
    data["Contribution_Max"].append(0.10)
    
    # Elizabeth Arden Bronzer in Asia/Face Make-up
    data["Portfolio"].append("Acme")
    data["Geography"].append("Asia")
    data["Category"].append("Face Make-up")
    data["Brand"].append("Elizabeth Arden")
    data["Segment"].append("Bronzer")
    data["Sales"].append(1200000)
    data["Margin"].append(0.30)
    data["Trend_Min"].append(0.05)
    data["Trend_Max"].append(0.10)
    data["Contribution_Min"].append(0.02)
    data["Contribution_Max"].append(0.10)
    
    # Add other segments to complete the structure
    # Add Fragrance in Europe
    data["Portfolio"].append("Acme")
    data["Geography"].append("Europe")
    data["Category"].append("Fragrance")
    data["Brand"].append("Kilian")
    data["Segment"].append("Perfume")
    data["Sales"].append(5000000)
    data["Margin"].append(0.40)
    data["Trend_Min"].append(-0.05)
    data["Trend_Max"].append(0.13)
    data["Contribution_Min"].append(0.02)
    data["Contribution_Max"].append(0.10)
    
    data["Portfolio"].append("Acme")
    data["Geography"].append("Europe")
    data["Category"].append("Fragrance")
    data["Brand"].append("Balmain")
    data["Segment"].append("Cologne")
    data["Sales"].append(4500000)
    data["Margin"].append(0.35)
    data["Trend_Min"].append(-0.05)
    data["Trend_Max"].append(0.13)
    data["Contribution_Min"].append(0.02)
    data["Contribution_Max"].append(0.10)
    
    data["Portfolio"].append("Acme")
    data["Geography"].append("Europe")
    data["Category"].append("Fragrance")
    data["Brand"].append("Frederic Malle")
    data["Segment"].append("Essential Oils")
    data["Sales"].append(3500000)
    data["Margin"].append(0.38)
    data["Trend_Min"].append(-0.03)
    data["Trend_Max"].append(0.04)
    data["Contribution_Min"].append(0.02)
    data["Contribution_Max"].append(0.10)
    
    # Add Hair Dye in Europe
    data["Portfolio"].append("Acme")
    data["Geography"].append("Europe")
    data["Category"].append("Hair Dye")
    data["Brand"].append("Brand X")
    data["Segment"].append("Hair Color")
    data["Sales"].append(7000000)
    data["Margin"].append(0.25)
    data["Trend_Min"].append(-0.05)
    data["Trend_Max"].append(0.13)
    data["Contribution_Min"].append(0.02)
    data["Contribution_Max"].append(0.10)
    
    # Add Make-up brushes in Europe
    data["Portfolio"].append("Acme")
    data["Geography"].append("Europe")
    data["Category"].append("Make-up brushes")
    data["Brand"].append("Brand Y")
    data["Segment"].append("Brushes")
    data["Sales"].append(3000000)
    data["Margin"].append(0.30)
    data["Trend_Min"].append(-0.05)
    data["Trend_Max"].append(0.13)
    data["Contribution_Min"].append(0.02)
    data["Contribution_Max"].append(0.10)
    
    # Add Face make-up in Europe
    data["Portfolio"].append("Acme")
    data["Geography"].append("Europe")
    data["Category"].append("Face make-up")
    data["Brand"].append("Brand Z")
    data["Segment"].append("Foundation")
    data["Sales"].append(6000000)
    data["Margin"].append(0.28)
    data["Trend_Min"].append(-0.05)
    data["Trend_Max"].append(0.13)
    data["Contribution_Min"].append(0.02)
    data["Contribution_Max"].append(0.05)
    
    # Add more categories in Asia
    data["Portfolio"].append("Acme")
    data["Geography"].append("Asia")
    data["Category"].append("Fragrance")
    data["Brand"].append("Kilian")
    data["Segment"].append("Perfume")
    data["Sales"].append(2000000)
    data["Margin"].append(0.42)
    data["Trend_Min"].append(-0.05)
    data["Trend_Max"].append(0.05)
    data["Contribution_Min"].append(0.01)
    data["Contribution_Max"].append(0.05)
    
    # Add North America and South America
    data["Portfolio"].append("Acme")
    data["Geography"].append("North America")
    data["Category"].append("Fragrance")
    data["Brand"].append("Kilian")
    data["Segment"].append("Perfume")
    data["Sales"].append(10000000)
    data["Margin"].append(0.40)
    data["Trend_Min"].append(-0.05)
    data["Trend_Max"].append(0.13)
    data["Contribution_Min"].append(0.00)
    data["Contribution_Max"].append(1.00)
    
    data["Portfolio"].append("Acme")
    data["Geography"].append("North America")
    data["Category"].append("Color Cosmetics")
    data["Brand"].append("Bobbi Brown")
    data["Segment"].append("Lipstick")
    data["Sales"].append(8000000)
    data["Margin"].append(0.45)
    data["Trend_Min"].append(-0.01)
    data["Trend_Max"].append(0.03)
    data["Contribution_Min"].append(0.00)
    data["Contribution_Max"].append(0.14)
    
    data["Portfolio"].append("Acme")
    data["Geography"].append("South America")
    data["Category"].append("Hair")
    data["Brand"].append("Aveda")
    data["Segment"].append("Shampoo")
    data["Sales"].append(3000000)
    data["Margin"].append(0.30)
    data["Trend_Min"].append(-0.15)
    data["Trend_Max"].append(0.05)
    data["Contribution_Min"].append(0.00)
    data["Contribution_Max"].append(1.00)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate contribution
    total_sales = df.groupby('Portfolio')['Sales'].transform('sum')
    df['Portfolio_Contribution'] = df.groupby('Portfolio')['Sales'].transform('sum') / total_sales
    
    # For other levels
    for i, level in enumerate(['Geography', 'Category', 'Brand']):
        parent_level = ['Portfolio', 'Geography', 'Category'][i]
        df[f'{level}_Contribution'] = df.groupby([parent_level, level])['Sales'].transform('sum') / df.groupby(parent_level)['Sales'].transform('sum')
    
    # Calculate segment contribution within brand
    df['Segment_Contribution'] = df.groupby(['Brand', 'Segment'])['Sales'].transform('sum') / df.groupby('Brand')['Sales'].transform('sum')
    
    return df

def create_multi_year_constraints():
    """
    Create constraints for multiple years based on the given requirements.
    """
    multi_year_constraints = {}
    
    # Year 1
    year_1 = {
        "Segment:Mascara": {"min_trend": 0.05, "max_trend": 0.05, "min_contribution": 0.05, "max_contribution": 0.05}
    }
    multi_year_constraints[1] = year_1
    
    # Year 2
    year_2 = {
        "Segment:Mascara": {"min_trend": 0.08, "max_trend": 0.08, "min_contribution": 0.08, "max_contribution": 0.08}
    }
    multi_year_constraints[2] = year_2
    
    # Year 3
    year_3 = {
        "Segment:Mascara": {"min_trend": 0.09, "max_trend": 0.09, "min_contribution": 0.09, "max_contribution": 0.09}
    }
    multi_year_constraints[3] = year_3
    
    # Year 4
    year_4 = {
        "Segment:Mascara": {"min_trend": 0.12, "max_trend": 0.12, "min_contribution": 0.12, "max_contribution": 0.12}
    }
    multi_year_constraints[4] = year_4
    
    # Year 5
    year_5 = {
        "Segment:Mascara": {"min_trend": 0.12, "max_trend": 0.12, "min_contribution": 0.12, "max_contribution": 0.12}
    }
    multi_year_constraints[5] = year_5
    
    return multi_year_constraints

def main():
    """
    Main function to demonstrate the Acme Optimizer
    """
    print("Creating Acme Optimizer...")
    
    # Create the optimizer
    optimizer = AcmeOptimizer()
    
    # Create the synthetic data based on requirements
    print("Creating synthetic dataset based on requirements...")
    df = create_synthetic_dataset_from_requirements()
    optimizer.df = df
    
    # Print the dataset summary
    print("\nDataset Summary:")
    print(f"Total number of records: {len(df)}")  # Add this line
    print(f"Total number of unique segments: {df['Segment'].nunique()}")
    print(f"Total sales: ${df['Sales'].sum():,.2f}")
    print(f"Total profit: ${(df['Sales'] * df['Margin']).sum():,.2f}")
    print(f"Average margin: {df['Margin'].mean():.2%}")
    
    # Create constraints based on the requirements
    print("\nCreating constraints based on the requirements...")
    constraints = {
        # Global constraints
        "Brand:Bobbi Brown": {"min_trend": -0.01, "max_trend": 0.03, "max_contribution": 0.14},
        "Category:Hair": {"max_contribution": 0.30},
        
        # Portfolio level constraints
        "Portfolio:Acme": {"min_trend": -0.05, "max_trend": 0.15},
        
        # Portfolio-specific constraints
        "Category:Skin/body": {"min_trend": 0.02, "max_trend": 0.03},
        "Category:Hair/APDO": {"max_contribution": 0.32},
        
        # Geography-specific constraints
        "Geography:North America": {
            "Category:Fragrance": {"min_contribution": 0.21, "max_contribution": 0.30},
            "Category:Color Cosmetics": {"min_contribution": 0.21, "max_contribution": 0.30}
        },
        "Geography:South America": {"min_trend": -0.15, "max_trend": 0.05},
        "Geography:Asia": {
            "max_contribution": 0.30,
            "Category:Fragrance": {"max_contribution": 0.05},
            "Category:Face Make-up": {
                "min_trend": -0.01, 
                "max_contribution": 0.05,
                "Brand:Bobbi Brown": {"min_contribution": 0.03},
                "Brand:Elizabeth Arden": {
                    "min_contribution": 0.08, 
                    "max_contribution": 0.14,
                    "max_trend": 0.07,
                    "Segment:Mascara": {
                        "min_trend": -0.08, 
                        "max_trend": -0.05, 
                        "min_contribution": 0.02, 
                        "max_contribution": 0.10
                    },
                    "Segment:Bronzer": {
                        "min_trend": 0.05, 
                        "max_trend": 0.10, 
                        "min_contribution": 0.02, 
                        "max_contribution": 0.10
                    }
                }
            }
        },
        "Geography:Europe": {
            "max_trend": 0.13,
            "Category:Face make-up": {"max_contribution": 0.05},
            "Category:Fragrance": {
                "max_contribution": 0.05,
                "Brand:Frederic Malle": {"min_trend": -0.03, "max_trend": 0.04}
            },
            "Category:Tools": {
                "min_contribution": 0.03, 
                "max_contribution": 0.07,
                "Brand:Bobbi Brown": {
                    "Segment:Lipstick": {
                        "min_trend": -0.04, 
                        "max_trend": 0.13, 
                        "min_contribution": 0.04, 
                        "max_contribution": 0.15
                    },
                    "Segment:Mascara": {
                        "min_trend": 0.02, 
                        "max_trend": 0.11, 
                        "min_contribution": 0.015, 
                        "max_contribution": 0.10
                    }
                },
                "Brand:Elizabeth Arden": {
                    "max_contribution": 0.13,
                    "Segment:Lipstick": {
                        "min_trend": 0.06, 
                        "max_trend": 0.22, 
                        "min_contribution": 0.03, 
                        "max_contribution": 0.06
                    },
                    "Segment:Toner": {
                        "min_trend": 0.085, 
                        "max_trend": 0.114, 
                        "min_contribution": 0.06, 
                        "max_contribution": 0.10
                    },
                    "Segment:Bronzer": {
                        "min_trend": -0.03, 
                        "max_trend": 0.14, 
                        "min_contribution": 0.01, 
                        "max_contribution": 0.03
                    },
                    "Segment:Mascara": {
                        "min_trend": 0.03, 
                        "max_trend": 0.16, 
                        "min_contribution": 0.06, 
                        "max_contribution": 0.12
                    }
                }
            }
        }
    }
    
    # Multi-year constraints for Mascara
    multi_year_constraints = create_multi_year_constraints()
    
    # Run the optimization for different objectives
    print("\nRunning optimization for different objectives...")
    
    # 1. Maximize Sales
    print("\n1. Maximize Sales")
    max_sales_results = optimizer.optimize_sales(constraints)
    print(f"Original Total Sales: ${max_sales_results['Sales'].sum():,.2f}")
    print(f"Optimized Total Sales: ${max_sales_results['New_Sales'].sum():,.2f}")
    print(f"Increase: ${max_sales_results['New_Sales'].sum() - max_sales_results['Sales'].sum():,.2f} ({(max_sales_results['New_Sales'].sum() / max_sales_results['Sales'].sum() - 1) * 100:.2f}%)")
    
    # 2. Maximize Margin
    print("\n2. Maximize Margin")
    max_margin_results = optimizer.optimize_margin(constraints)
    print(f"Original Total Profit: ${(max_margin_results['Sales'] * max_margin_results['Margin']).sum():,.2f}")
    print(f"Optimized Total Profit: ${max_margin_results['New_Profit'].sum():,.2f}")
    print(f"Increase: ${max_margin_results['New_Profit'].sum() - (max_margin_results['Sales'] * max_margin_results['Margin']).sum():,.2f} ({(max_margin_results['New_Profit'].sum() / (max_margin_results['Sales'] * max_margin_results['Margin']).sum() - 1) * 100:.2f}%)")
    
    # 3. Hit sales target while maximizing margin
    target_sales = df['Sales'].sum() * 1.1  # 10% growth target
    print(f"\n3. Hit Sales Target of ${target_sales:,.2f} while Maximizing Margin")
    sales_target_results = optimizer.hit_sales_target_maximize_margin(target_sales, constraints)
    print(f"Original Total Sales: ${sales_target_results['Sales'].sum():,.2f}")
    print(f"Target Sales: ${target_sales:,.2f}")
    print(f"Achieved Sales: ${sales_target_results['New_Sales'].sum():,.2f}")
    print(f"Original Total Profit: ${(sales_target_results['Sales'] * sales_target_results['Margin']).sum():,.2f}")
    print(f"Optimized Total Profit: ${sales_target_results['New_Profit'].sum():,.2f}")
    print(f"Profit Increase: ${sales_target_results['New_Profit'].sum() - (sales_target_results['Sales'] * sales_target_results['Margin']).sum():,.2f} ({(sales_target_results['New_Profit'].sum() / (sales_target_results['Sales'] * sales_target_results['Margin']).sum() - 1) * 100:.2f}%)")
    
    # 4. Hit margin target while maximizing sales
    target_margin = (df['Sales'] * df['Margin']).sum() * 1.15  # 15% profit growth target
    print(f"\n4. Hit Margin Target of ${target_margin:,.2f} while Maximizing Sales")
    margin_target_results = optimizer.hit_margin_target_maximize_sales(target_margin, constraints)
    print(f"Original Total Profit: ${(margin_target_results['Sales'] * margin_target_results['Margin']).sum():,.2f}")
    print(f"Target Profit: ${target_margin:,.2f}")
    print(f"Achieved Profit: ${margin_target_results['New_Profit'].sum():,.2f}")
    print(f"Sales Growth: ${margin_target_results['New_Sales'].sum() - margin_target_results['Sales'].sum():,.2f} ({(margin_target_results['New_Sales'].sum() / margin_target_results['Sales'].sum() - 1) * 100:.2f}%)")

    # 5. Five-year projections
    print("\n5. Five-year Projections")
    years = 5
    
    # Run different scenarios for 5 years
    print("\nRunning 5-year scenarios...")
    
    # Maximize Sales over 5 years
    max_sales_5y = optimizer.optimize_sales(constraints, years=years)
    print("\n5-year Sales Maximization:")
    for year in range(1, years + 1):
        results = max_sales_5y[year]
        print(f"\nYear {year}:")
        print(f"Total Sales: ${results['New_Sales'].sum():,.2f}")
        print(f"Total Profit: ${results['New_Profit'].sum():,.2f}")
        print(f"Average Growth: {results['New_Trend'].mean():.2%}")

    # Maximize Margin over 5 years
    max_margin_5y = optimizer.optimize_margin(constraints, years=years)
    print("\n5-year Margin Maximization:")
    for year in range(1, years + 1):
        results = max_margin_5y[year]
        print(f"\nYear {year}:")
        print(f"Total Sales: ${results['New_Sales'].sum():,.2f}")
        print(f"Total Profit: ${results['New_Profit'].sum():,.2f}")
        print(f"Average Margin: {(results['New_Profit'].sum() / results['New_Sales'].sum()):.2%}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. Sales Maximization Visualization
    fig_sales = optimizer.visualize_results(max_sales_results, metric='Sales', level='Geography')
    plt.savefig('sales_optimization.png')
    plt.close()

    # 2. Margin Maximization Visualization
    fig_margin = optimizer.visualize_results(max_margin_results, metric='Profit', level='Geography')
    plt.savefig('margin_optimization.png')
    plt.close()

    # 3. Five-year Trend Visualization
    plt.figure(figsize=(12, 6))
    years_range = range(1, years + 1)
    
    # Plot sales trend
    sales_trend = [max_sales_5y[year]['New_Sales'].sum() / 1e6 for year in years_range]
    plt.plot(years_range, sales_trend, 'b-', label='Sales Maximization')
    
    # Plot margin trend
    margin_trend = [max_margin_5y[year]['New_Profit'].sum() / 1e6 for year in years_range]
    plt.plot(years_range, margin_trend, 'r-', label='Margin Maximization')
    
    plt.xlabel('Year')
    plt.ylabel('Amount (Millions $)')
    plt.title('5-Year Projection Comparison')
    
    # Add value labels to the points
    for i, (sales, margin) in enumerate(zip(sales_trend, margin_trend), 1):
        plt.text(i, sales, f'${sales:.1f}M', ha='left', va='bottom')
        plt.text(i, margin, f'${margin:.1f}M', ha='right', va='top')
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('five_year_projection.png')
    plt.close()

    # Generate detailed reports
    print("\nGenerating detailed reports...")
    
    # Generate reports for each optimization scenario
    sales_reports = optimizer.generate_reports(max_sales_results)
    margin_reports = optimizer.generate_reports(max_margin_results)
    sales_target_reports = optimizer.generate_reports(sales_target_results)
    margin_target_reports = optimizer.generate_reports(margin_target_results)
    
    # Save reports to JSON files
    with open('sales_optimization_report.json', 'w') as f:
        json.dump(sales_reports, f, indent=4)
    
    with open('margin_optimization_report.json', 'w') as f:
        json.dump(margin_reports, f, indent=4)
    
    with open('sales_target_report.json', 'w') as f:
        json.dump(sales_target_reports, f, indent=4)
    
    with open('margin_target_report.json', 'w') as f:
        json.dump(margin_target_reports, f, indent=4)

    print("\nOptimization complete. Reports and visualizations have been saved.")

if __name__ == "__main__":
    main()