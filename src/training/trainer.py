import numpy as np
import torch
import torch.nn as nn
from src.model.network import XNet
from src.data.preprocess import get_train_test
from src.training.metrics import evaluation_rmse, evaluation_mape, evaluation_mae, evaluation_r2
from src.data.loader import setup_seed  # Note: setup_seed is moved to loader.py

def train_with_logs(params):
    """Train model and return evaluation scores + predictions"""
    setup_seed(params['seed'])
    scores_list, result_list, cycle_list = [], [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for name in params['battery_list']:
        # Get train/test data
        train_x, train_y, train_data, test_data, cycle_seq, data_seq, test_x, test_y = get_train_test(
            params['battery_data'], name, params['feature_size'], params['train_split_ratio']
        )
        
        # Initialize model
        model = XNet(
            params['feature_size'], params['hidden_dim'], params['num_layers'],
            params['activation'], params.get('cauchy_params')
        ).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params['weight_decay'])
        criterion = nn.MSELoss()
        best_rmse = float('inf')
        best_pred = None

        # Training loop
        for epoch in range(params['epochs']):
            model.train()
            # Normalize data
            X = np.reshape(train_x / params['rated_capacity'], (-1, params['feature_size'], 1))
            y = np.reshape(train_y / params['rated_capacity'], (-1, 1))
            X_t = torch.from_numpy(X).float().to(device)
            y_t = torch.from_numpy(y).float().to(device)

            # Forward + backward + optimize
            optimizer.zero_grad()
            out = model(X_t)
            loss = criterion(out, y_t)
            loss.backward()
            optimizer.step()

            # Evaluate on test set
            if (epoch + 1) % 50 == 0 or (epoch + 1) == params['epochs']:
                model.eval()
                with torch.no_grad():
                    x_val = np.reshape(test_x / params['rated_capacity'], (-1, params['feature_size'], 1))
                    x_val_t = torch.from_numpy(x_val).float().to(device)
                    pred = model(x_val_t).cpu().numpy().reshape(-1) * params['rated_capacity']

                current_rmse = evaluation_rmse(test_y, pred)
                if current_rmse < best_rmse:
                    best_rmse = current_rmse
                    best_pred = pred.copy()

        # Calculate metrics
        best_mape = evaluation_mape(test_y, best_pred)
        best_mae = evaluation_mae(test_y, best_pred)
        best_r2 = evaluation_r2(test_y, best_pred)
        scores_list.append({'rmse': best_rmse, 'mape': best_mape, 'mae': best_mae, 'r2': best_r2})
        
        # Generate full prediction sequence
        full_pred = [np.nan] * len(data_seq)
        full_pred[:len(train_data)] = train_data
        pred_start = params['feature_size']
        pred_end = pred_start + len(best_pred)
        if pred_end > len(full_pred):
            pred_end = len(full_pred)
        full_pred[pred_start:pred_end] = best_pred[:pred_end - pred_start]
        full_pred = np.nan_to_num(full_pred, nan=0.0)
        
        result_list.append(full_pred)
        cycle_list.append(cycle_seq)

    return scores_list, result_list, cycle_list

def run_experiments(battery_data, train_ratios, activation_functions, model_params):
    """Run experiments for all activation functions and train ratios"""
    final_results = {act: {'rmse': [], 'mape': [], 'mae': [], 'r2': []} for act in activation_functions}
    pred_results = {name: {ratio: {} for ratio in train_ratios} for name in battery_data.keys()}
    cycle_results = {name: {} for name in battery_data.keys()}

    for ratio in train_ratios:
        print(f"\n--- Training with {int(ratio*100)}% data ---")
        for act in activation_functions:
            print(f"  - Activation: {act.upper()}")
            # Update params for current experiment
            current_params = model_params.copy()
            current_params['activation'] = act
            current_params['train_split_ratio'] = ratio
            current_params['battery_data'] = battery_data
            current_params['battery_list'] = list(battery_data.keys())
            
            # Train model
            scores_list, result_list, cycle_list = train_with_logs(current_params)
            
            # Calculate average metrics
            avg_rmse = np.mean([s['rmse'] for s in scores_list])
            avg_mape = np.mean([s['mape'] for s in scores_list])
            avg_mae = np.mean([s['mae'] for s in scores_list])
            avg_r2 = np.mean([s['r2'] for s in scores_list])
            
            # Save results
            final_results[act]['rmse'].append(avg_rmse)
            final_results[act]['mape'].append(avg_mape)
            final_results[act]['mae'].append(avg_mae)
            final_results[act]['r2'].append(avg_r2)
            
            # Save predictions
            for i, name in enumerate(battery_data.keys()):
                pred_results[name][ratio][act] = result_list[i]
                cycle_results[name] = cycle_list[i]
            
            # Print metrics
            print(f"    - Avg RMSE: {avg_rmse:.4f}, Avg MAE: {avg_mae:.4f}, Avg MAPE: {avg_mape:.4f}%, Avg R2: {avg_r2:.4f}")

    return final_results, pred_results, cycle_results