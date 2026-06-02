import os
import pandas as pd
import itertools
import subprocess
import json
from datetime import datetime
import argparse
import glob
import time
import shutil

class AttackGridRunner:
    def __init__(self, base_output_dir="attack_results_grid"):
        self.base_output_dir = base_output_dir
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def generate_parameter_grid(self):
        
        # datasets = ['sst', 'qnli', 'rte']
        
        betas = [1.0, 3.0]  
        lambdas = [0.1, 1.0]
        
        constraints = [
            # {'name': 'early_stopping', 'early_stop': True, 'lev_max': None},
            {'name': 'lev_3_stop', 'early_stop': True, 'lev_max': 3, 'ascii': True},
            {'name': 'lev_3_stop', 'early_stop': True, 'lev_max': 3, 'ascii': False}
            # {'name': 'lev_3', 'early_stop': False, 'lev_max': 3}
            # {'name': 'lev_7', 'early_stop': False, 'lev_max': 7}
        ]
        
        experiments = []
        exp_id = 0
        
        # for dataset in datasets:
        for beta in betas:
            for lambda_lev in lambdas:
                for constraint in constraints:
                    experiments.append({
                        'exp_id': exp_id,
                        # 'dataset': dataset,
                        'beta': beta,
                        'lambda_lev': lambda_lev,
                        'ascii': constraint['ascii'],
                        'lev_max': constraint['lev_max'],
                        'early_stop': constraint['early_stop'],
                        'constraint_name': constraint['name']
                    })
                    exp_id += 1
        
        print(f"Total experiments to run: {len(experiments)}")
        # print(f"  - Datasets: {len(datasets)}")
        print(f"  - Betas: {len(betas)}")
        print(f"  - Lambdas: {len(lambdas)}")
        print(f"  - Constraints: {len(constraints)}")
        print(f"  = {len(betas)} × {len(lambdas)} × {len(constraints)} = {len(experiments)}")
        
        return experiments

    def run_experiment(self, exp):
        exp_dir = os.path.join(
            self.base_output_dir,
            self.timestamp,
            # exp['dataset'],
            f"beta_{exp['beta']}",
            f"lambda_{exp['lambda_lev']}",
            exp['constraint_name']
        )
        os.makedirs(exp_dir, exist_ok=True)
        
        suffix = f"_beta_{exp['beta']}_lambda_{exp['lambda_lev']}_{exp['constraint_name']}"
        
        cmd = [
            'python3', 'attack.py',
            # '--dataset', exp['dataset'],
            '--dataset', 'sst',
            '--attack_name', 'our_method',
            '--beta', str(exp['beta']),
            '--lambda_lev', str(exp['lambda_lev']),
            '--max_steps', '100',
            '--model_name', 'meta-llama/Llama-3.2-1B-Instruct',
            '--loss', 'margin',
            '--tau', '0',
            '--p_ins', '0.05',
            '--p_del', '0.05',
            '--p_sub', '0.90',
            '--size', '200',
            '--device', 'privateuseone:0',
            '--sufix', suffix
        ]
        
        if exp['early_stop']:
            cmd.append('--early_stop') 
        
        if exp['lev_max'] is not None:
            cmd.extend(['--lev_max', str(exp['lev_max'])])

        if exp['ascii']:
            cmd.append('--ascii')     
        
        print(f"\nRunning: {' '.join(cmd)}")
        
        output_file = os.path.join(exp_dir, f"output.log")
        result_file = os.path.join(exp_dir, f"results.csv")
        mcmc_results_file = os.path.join(exp_dir, f"mcmc_results.csv")  # Добавляем файл для MCMC результатов
        
        try:
            start_time = time.time()
            with open(output_file, 'w') as f:
                process = subprocess.run(cmd, capture_output=True, text=True)
                f.write("=== COMMAND ===\n")
                f.write(' '.join(cmd) + "\n\n")
                f.write("=== STDOUT ===\n")
                f.write(process.stdout)
                if process.stderr:
                    f.write("\n=== STDERR ===\n")
                    f.write(process.stderr)
            
            time.sleep(1)
            
            # Ищем файл результатов в разных местах
            result_paths = []
            mcmc_paths = []
            
            # 1. В стандартной папке results_attack
            results_dir = f"results_attack/llm_classifier/sst/Llama-3.2-1B-Instruct"
            if os.path.exists(results_dir):
                pattern = f"*{suffix}*.csv"
                matching_files = glob.glob(os.path.join(results_dir, pattern))
                result_paths.extend(matching_files)
                print(f"Search in {results_dir}: found {len(matching_files)} files")
            
            # 2. В папке mcmc_results
            mcmc_dir = "mcmc_results"
            if os.path.exists(mcmc_dir):
                # Ищем файлы MCMC с соответствующими параметрами
                mcmc_pattern = f"*beta_{exp['beta']}_lambda_{exp['lambda_lev']}*"
                mcmc_matching = glob.glob(os.path.join(mcmc_dir, mcmc_pattern))
                mcmc_paths.extend(mcmc_matching)
                print(f"Search in {mcmc_dir}: found {len(mcmc_matching)} MCMC files")
            
            # 3. В текущей директории
            matching_files = glob.glob(f"*{suffix}*.csv")
            result_paths.extend(matching_files)
            
            # 4. В папке эксперимента
            matching_files = glob.glob(os.path.join(exp_dir, f"*{suffix}*.csv"))
            result_paths.extend(matching_files)
            
            # 5. Во всех поддиректориях results_attack
            for root, dirs, files in os.walk('results_attack'):
                for file in files:
                    if suffix in file and file.endswith('.csv'):
                        result_paths.append(os.path.join(root, file))
                    # Также ищем MCMC файлы
                    if 'all_iters' in file and f"beta_{exp['beta']}" in file and f"lambda_{exp['lambda_lev']}" in file:
                        mcmc_paths.append(os.path.join(root, file))
            
            # 6. Поиск MCMC файлов в корневой директории
            if os.path.exists('mcmc_results'):
                for file in os.listdir('mcmc_results'):
                    if file.endswith('.csv') and f"beta_{exp['beta']}" in file and f"lambda_{exp['lambda_lev']}" in file:
                        mcmc_paths.append(os.path.join('mcmc_results', file))
            
            # Удаляем дубликаты
            result_paths = list(set(result_paths))
            mcmc_paths = list(set(mcmc_paths))
            
            # Копируем основной файл результатов
            if result_paths:
                latest_file = max(result_paths, key=os.path.getctime)
                shutil.copy(latest_file, result_file)
                print(f"✓ Found result file: {latest_file}")
                print(f"✓ Copied to: {result_file}")
                
                elapsed_time = time.time() - start_time
                
                # Копируем MCMC результаты если они есть
                mcmc_copied = False
                if mcmc_paths:
                    latest_mcmc = max(mcmc_paths, key=os.path.getctime)
                    shutil.copy(latest_mcmc, mcmc_results_file)
                    print(f"✓ Found MCMC file: {latest_mcmc}")
                    print(f"✓ Copied to: {mcmc_results_file}")
                    mcmc_copied = True
                
                return {
                    'success': True,
                    'exp_id': exp['exp_id'],
                    'beta': exp['beta'],
                    'lambda_lev': exp['lambda_lev'],
                    'constraint': exp['constraint_name'],
                    'lev_max': exp['lev_max'],
                    'early_stop': exp['early_stop'],
                    'output_file': output_file,
                    'result_file': result_file,
                    'mcmc_results_file': mcmc_results_file if mcmc_copied else None,
                    'elapsed_time': elapsed_time
                }
            else:
                print(f"⚠ No result file found for suffix: {suffix}")
                print(f"  Checked locations:")
                print(f"    - results_attack/llm_classifier/sst/Llama-3.2-1B-Instruct/")
                print(f"    - Current directory")
                print(f"    - {exp_dir}")
                
                # Выводим содержимое директории для отладки
                if os.path.exists(results_dir):
                    print(f"  Files in {results_dir}:")
                    for f in os.listdir(results_dir):
                        print(f"    - {f}")
                
                return {
                    'success': False,
                    'exp_id': exp['exp_id'],
                    'error': 'Result file not found',
                    'suffix': suffix,
                    'output_file': output_file
                }
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return {'success': False, 'exp_id': exp['exp_id'], 'error': str(e)}

    def run_all_experiments(self):
        experiments = self.generate_parameter_grid()
        
        print(f"Total experiments: {len(experiments)}")
        print(f"Output directory: {os.path.join(self.base_output_dir, self.timestamp)}")
        
        results = []
        
        for i, exp in enumerate(experiments):
            print(f"\n{'='*80}")
            print(f"Experiment {i+1}/{len(experiments)}")
            # print(f"  Dataset: {exp['dataset']}")
            print(f"  Beta: {exp['beta']}")
            print(f"  Lambda: {exp['lambda_lev']}")
            print(f"  Constraint: {exp['constraint_name']} (lev_max={exp['lev_max']}, early_stop={exp['early_stop']})")
            print(f"{'='*80}")
            
            result = self.run_experiment(exp)
            results.append(result)
            
            # Сохраняем прогресс после каждого эксперимента
            self.save_progress(results)
        
        # Финальная статистика
        successful = sum(1 for r in results if r.get('success', False))
        print(f"\n{'='*80}")
        print(f"ALL EXPERIMENTS COMPLETED")
        print(f"Successful: {successful}/{len(results)}")
        print(f"Failed: {len(results) - successful}")
        print(f"{'='*80}")
        
        return results

    def save_progress(self, results):
        progress_file = os.path.join(
            self.base_output_dir,
            self.timestamp,
            f"experiment_progress.csv"
        )
        os.makedirs(os.path.dirname(progress_file), exist_ok=True)
        
        df = pd.DataFrame(results)
        df.to_csv(progress_file, index=False)
        print(f"Progress saved to {progress_file} ({len(results)} experiments)")

def main():
    parser = argparse.ArgumentParser(description='Run attack experiments with parameter grid')
    parser.add_argument('--max_experiments', type=int, default=None,
                       help='Maximum number of experiments to run (for testing)')
    args = parser.parse_args()
    
    runner = AttackGridRunner()
    
    if args.max_experiments:
        experiments = runner.generate_parameter_grid()[:args.max_experiments]
        print(f"Running only {len(experiments)} experiments (limited by --max_experiments)")
        
        results = []
        for i, exp in enumerate(experiments):
            result = runner.run_experiment(exp)
            results.append(result)
            runner.save_progress(results)
    else:
        results = runner.run_all_experiments()

if __name__ == '__main__':
    main()


# import os
# import pandas as pd
# import subprocess
# import shutil
# import glob
# from datetime import datetime

# class AttackGridResumer:
#     def __init__(self, base_output_dir="attack_results_grid"):
#         self.base_output_dir = base_output_dir
#         # Укажите timestamp вашего предыдущего запуска
#         self.previous_timestamp = "20260529_014701"  # ЗАМЕНИТЕ на ваш реальный timestamp
#         self.new_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

#     def get_completed_experiments(self):
#         """Получает список уже выполненных экспериментов из предыдущего запуска"""
#         progress_file = os.path.join(
#             self.base_output_dir,
#             self.previous_timestamp,
#             "experiment_progress.csv"
#         )
        
#         if os.path.exists(progress_file):
#             df = pd.read_csv(progress_file)
#             completed = []
#             for _, row in df.iterrows():
#                 if row.get('success', False):
#                     completed.append({
#                         'beta': row['beta'],
#                         'lambda_lev': row['lambda_lev'],
#                         'constraint': row['constraint']
#                     })
#             print(f"Found {len(completed)} completed experiments")
#             return completed
#         return []
    
#     def generate_missing_experiments(self):
#         """Генерирует только недостающие эксперименты"""
#         completed = self.get_completed_experiments()
        
#         # Все возможные комбинации
#         all_experiments = []
#         betas = [1.0, 3.0]
#         lambdas = [0.1, 1.0]
#         constraints = [
#             {'name': 'early_stopping', 'early_stop': True, 'lev_max': None},
#             {'name': 'lev_3', 'early_stop': False, 'lev_max': 3}
#         ]
        
#         exp_id = 0
#         for beta in betas:
#             for lambda_lev in lambdas:
#                 for constraint in constraints:
#                     exp = {
#                         'exp_id': exp_id,
#                         'beta': beta,
#                         'lambda_lev': lambda_lev,
#                         'constraint_name': constraint['name'],
#                         'lev_max': constraint['lev_max'],
#                         'early_stop': constraint['early_stop']
#                     }
                    
#                     # Проверяем, выполнен ли этот эксперимент
#                     is_completed = any(
#                         c['beta'] == beta and 
#                         c['lambda_lev'] == lambda_lev and 
#                         c['constraint'] == constraint['name']
#                         for c in completed
#                     )
                    
#                     if not is_completed:
#                         all_experiments.append(exp)
#                         print(f"Missing: beta={beta}, lambda={lambda_lev}, constraint={constraint['name']}")
                    
#                     exp_id += 1
        
#         print(f"\nMissing experiments: {len(all_experiments)}")
#         return all_experiments
    
#     def run_missing_experiments(self):
#         """Запускает только недостающие эксперименты"""
#         experiments = self.generate_missing_experiments()
        
#         if not experiments:
#             print("All experiments are already completed!")
#             return []
        
#         results = []
        
#         for i, exp in enumerate(experiments):
#             print(f"\n{'='*80}")
#             print(f"Running missing experiment {i+1}/{len(experiments)}")
#             print(f"  Beta: {exp['beta']}")
#             print(f"  Lambda: {exp['lambda_lev']}")
#             print(f"  Constraint: {exp['constraint_name']}")
#             print(f"{'='*80}")
            
#             result = self.run_single_experiment(exp)
#             results.append(result)
            
#             # Сохраняем прогресс в новой папке (или дописываем в старую)
#             self.save_progress(results, append_to_previous=True)
        
#         return results
    
#     def run_single_experiment(self, exp):
#         """Запускает один эксперимент"""
#         exp_dir = os.path.join(
#             self.base_output_dir,
#             self.previous_timestamp,  # Используем папку предыдущего запуска
#             f"beta_{exp['beta']}",
#             f"lambda_{exp['lambda_lev']}",
#             exp['constraint_name']
#         )
#         os.makedirs(exp_dir, exist_ok=True)
        
#         suffix = f"_beta_{exp['beta']}_lambda_{exp['lambda_lev']}_{exp['constraint_name']}"
        
#         cmd = [
#             'python', 'attack.py',
#             '--dataset', 'sst',
#             '--attack_name', 'our_method',
#             '--beta', str(exp['beta']),
#             '--lambda_lev', str(exp['lambda_lev']),
#             '--max_steps', '100',
#             '--model_name', 'meta-llama/Llama-3.2-1B-Instruct',
#             '--loss', 'margin',
#             '--tau', '0',
#             '--ascii', 'True',
#             '--p_ins', '0.05',
#             '--p_del', '0.05',
#             '--p_sub', '0.90',
#             '--size', '200',
#             '--device', 'xpu',
#             '--sufix', suffix
#         ]
        
#         if exp['early_stop']:
#             cmd.append('--early_stop')
        
#         if exp['lev_max'] is not None:
#             cmd.extend(['--lev_max', str(exp['lev_max'])])
        
#         print(f"Running: {' '.join(cmd)}")
        
#         output_file = os.path.join(exp_dir, f"output_resume.log")
#         result_file = os.path.join(exp_dir, f"results.csv")
#         mcmc_results_file = os.path.join(exp_dir, f"mcmc_results.csv")  # Добавляем файл для MCMC результатов
        
#         try:
#             process = subprocess.run(cmd, capture_output=True, text=True)
            
#             with open(output_file, 'w') as f:
#                 f.write("=== COMMAND ===\n")
#                 f.write(' '.join(cmd) + "\n\n")
#                 f.write("=== STDOUT ===\n")
#                 f.write(process.stdout)
#                 if process.stderr:
#                     f.write("\n=== STDERR ===\n")
#                     f.write(process.stderr)
            
#             # Поиск файлов результатов (как в вашем оригинальном коде)
#             result_paths = []
#             mcmc_paths = []
            
#             # 1. В стандартной папке results_attack
#             results_dir = f"results_attack/llm_classifier/sst/Llama-3.2-1B-Instruct"
#             if os.path.exists(results_dir):
#                 pattern = f"*{suffix}*.csv"
#                 matching_files = glob.glob(os.path.join(results_dir, pattern))
#                 result_paths.extend(matching_files)
#                 print(f"Search in {results_dir}: found {len(matching_files)} files")
            
#             # 2. В папке mcmc_results
#             mcmc_dir = "mcmc_results"
#             if os.path.exists(mcmc_dir):
#                 # Ищем файлы MCMC с соответствующими параметрами
#                 mcmc_pattern = f"*beta_{exp['beta']}_lambda_{exp['lambda_lev']}*"
#                 mcmc_matching = glob.glob(os.path.join(mcmc_dir, mcmc_pattern))
#                 mcmc_paths.extend(mcmc_matching)
#                 print(f"Search in {mcmc_dir}: found {len(mcmc_matching)} MCMC files")
            
#             # 3. В текущей директории
#             matching_files = glob.glob(f"*{suffix}*.csv")
#             result_paths.extend(matching_files)
            
#             # 4. В папке эксперимента
#             matching_files = glob.glob(os.path.join(exp_dir, f"*{suffix}*.csv"))
#             result_paths.extend(matching_files)
            
#             # 5. Во всех поддиректориях results_attack
#             for root, dirs, files in os.walk('results_attack'):
#                 for file in files:
#                     if suffix in file and file.endswith('.csv'):
#                         result_paths.append(os.path.join(root, file))
#                     # Также ищем MCMC файлы
#                     if 'all_iters' in file and f"beta_{exp['beta']}" in file and f"lambda_{exp['lambda_lev']}" in file:
#                         mcmc_paths.append(os.path.join(root, file))
            
#             # 6. Поиск MCMC файлов в корневой директории
#             if os.path.exists('mcmc_results'):
#                 for file in os.listdir('mcmc_results'):
#                     if file.endswith('.csv') and f"beta_{exp['beta']}" in file and f"lambda_{exp['lambda_lev']}" in file:
#                         mcmc_paths.append(os.path.join('mcmc_results', file))
            
#             # Удаляем дубликаты
#             result_paths = list(set(result_paths))
#             mcmc_paths = list(set(mcmc_paths))
            
#             # Копируем основной файл результатов
#             if result_paths:
#                 latest_file = max(result_paths, key=os.path.getctime)
#                 shutil.copy(latest_file, result_file)
#                 print(f"✓ Result saved to {result_file}")
                
#                 # Копируем MCMC результаты если они есть
#                 mcmc_copied = False
#                 if mcmc_paths:
#                     latest_mcmc = max(mcmc_paths, key=os.path.getctime)
#                     shutil.copy(latest_mcmc, mcmc_results_file)
#                     print(f"✓ MCMC results saved to {mcmc_results_file}")
#                     mcmc_copied = True
                
#                 return {
#                     'success': True,
#                     'beta': exp['beta'],
#                     'lambda_lev': exp['lambda_lev'],
#                     'constraint': exp['constraint_name'],
#                     'lev_max': exp['lev_max'],
#                     'early_stop': exp['early_stop'],
#                     'mcmc_copied': mcmc_copied,
#                     **exp
#                 }
#             else:
#                 print(f"⚠ No result file found for suffix: {suffix}")
#                 return {'success': False, 'error': 'Result file not found', **exp}
            
#         except Exception as e:
#             print(f"✗ Error: {e}")
#             return {'success': False, 'error': str(e), **exp}
    
#     def save_progress(self, results, append_to_previous=True):
#         """Сохраняет прогресс"""
#         if append_to_previous and os.path.exists(
#             os.path.join(self.base_output_dir, self.previous_timestamp, "experiment_progress.csv")
#         ):
#             # Загружаем старый прогресс
#             old_progress_file = os.path.join(
#                 self.base_output_dir, self.previous_timestamp, "experiment_progress.csv"
#             )
#             old_df = pd.read_csv(old_progress_file)
            
#             # Добавляем новые результаты
#             new_df = pd.DataFrame(results)
#             combined_df = pd.concat([old_df, new_df], ignore_index=True)
            
#             # Сохраняем обновленный прогресс
#             combined_df.to_csv(old_progress_file, index=False)
#             print(f"Progress appended to {old_progress_file}")
#         else:
#             # Создаем новый файл
#             progress_file = os.path.join(
#                 self.base_output_dir,
#                 self.previous_timestamp,
#                 f"resume_progress_{self.new_timestamp}.csv"
#             )
#             df = pd.DataFrame(results)
#             df.to_csv(progress_file, index=False)
#             print(f"Progress saved to {progress_file}")

# def main():
#     resumer = AttackGridResumer()
#     # Установите правильный timestamp
#     resumer.previous_timestamp = input("Enter the timestamp of previous run (e.g., 20241125_143000): ")
    
#     results = resumer.run_missing_experiments()
    
#     successful = sum(1 for r in results if r.get('success', False))
#     mcmc_copied = sum(1 for r in results if r.get('success', False) and r.get('mcmc_copied', False))
    
#     print(f"\n{'='*80}")
#     print(f"RESUME COMPLETED")
#     print(f"Successfully ran: {successful}/{len(results)}")
#     print(f"MCMC files copied: {mcmc_copied}/{successful}")
#     print(f"{'='*80}")

# if __name__ == '__main__':
#     main()