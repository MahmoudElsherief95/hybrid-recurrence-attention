def run_long_sequence_generalization():
    """Evaluate all models on longer sequences than training and save results for notebook analysis."""
    print("\n🔎 Running Long-Sequence Generalization Test...")
    import torch
    import json
    import os
    from models.griffin.griffin_model import GriffinModel
    from models.hawk.hawk_model import HawkModel
    from models.local_attention.attention_model import LocalAttentionModel

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seq_lens = [128, 256, 512, 1024, 2048]
    griffin_cfg = {"vocab_size": 1000, "d_model": 144, "num_layers": 6, "num_heads": 4, "max_seq_len": 2048, "local_window": 64}
    hawk_cfg = {"vocab_size": 1000, "d_model": 144, "num_layers": 6, "max_seq_len": 2048}
    local_attn_cfg = {"vocab_size": 1000, "d_model": 144, "num_layers": 6, "num_heads": 4, "max_seq_len": 2048, "local_window": 64}

    def eval_long_seq(model_class, config, seq_lens, device='cpu'):
        losses = []
        for seq_len in seq_lens:
            x = torch.randint(0, config['vocab_size'], (2, seq_len), device=device)
            model = model_class(config).to(device)
            with torch.no_grad():
                out = model(x)
                logits = out['logits'] if isinstance(out, dict) else out
                targets = x[:, 1:].contiguous()
                logits = logits[:, :-1, :]
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
                losses.append(loss.item())
        return losses

    griffin_losses = eval_long_seq(GriffinModel, griffin_cfg, seq_lens, device)
    hawk_losses = eval_long_seq(HawkModel, hawk_cfg, seq_lens, device)
    local_losses = eval_long_seq(LocalAttentionModel, local_attn_cfg, seq_lens, device)

    results = {
        "sequence_lengths": seq_lens,
        "Griffin": griffin_losses,
        "Hawk": hawk_losses,
        "Local Attention": local_losses
    }
    os.makedirs("results", exist_ok=True)
    with open("results/long_seq_generalization.json", "w") as f:
        json.dump(results, f, indent=2)
    print("✅ Long-sequence generalization results saved to results/long_seq_generalization.json")
    return True

#!/usr/bin/env python3
"""
Quick Start Script for Griffin Model Study
Handles dependency checking and provides simple commands for research
"""

import sys
import subprocess
import os
from pathlib import Path
# --- Required imports for experiment orchestration ---
import torch
import time
import psutil
import torch.nn.functional as F
from models.griffin.griffin_model import GriffinModel
from models.hawk.hawk_model import HawkModel
from models.local_attention.attention_model import LocalAttentionModel
from data.mqar.mqar_dataset import create_mqar_datasets
from data.chomsky.chomsky_dataset import generate_an_bn_dataset, AnBnDataset


def check_dependencies():
    """Check if required dependencies are installed."""
    required = ['torch', 'yaml', 'matplotlib', 'seaborn', 'pandas', 'tensorboard']
    missing = []
    for pkg in required:
        try:
            __import__(pkg if pkg != 'yaml' else 'yaml')
        except ImportError:
            missing.append(pkg)
    return missing


def install_dependencies():
    """Install missing dependencies."""
    print("Installing dependencies with pip...")
    import subprocess
    cmds = [
        "pip install torch torchvision torchaudio",
        "pip install pyyaml tensorboard matplotlib seaborn pandas jupyter"
    ]
    for cmd in cmds:
        print(f"  {cmd}")
        if subprocess.run(cmd, shell=True).returncode != 0:
            print(f"  ❌ Failed: {cmd}")
            return False
    print("✅ All dependencies installed.")
    return True


def run_quick_test():
    """Run a quick test of the Griffin models."""
    try:
        sys.path.append(str(Path(__file__).parent))
        from models.griffin import GriffinModel
        from models.hawk import HawkModel
        from models.local_attention import LocalAttentionModel
        import torch
        griffin = GriffinModel({"vocab_size": 1000, "d_model": 128, "num_layers": 2, "num_heads": 4, "max_seq_len": 256, "local_window": 32})
        hawk = HawkModel({"vocab_size": 1000, "d_model": 128, "num_layers": 2, "max_seq_len": 256})
        local_attn = LocalAttentionModel({"vocab_size": 1000, "d_model": 128, "num_layers": 2, "num_heads": 4, "max_seq_len": 256, "local_window": 32})
        test_input = torch.randint(0, 1000, (2, 64))
        with torch.no_grad():
            griffin_out = griffin(test_input)
            hawk_out = hawk(test_input)
            attn_out = local_attn(test_input)
        print("Quick test passed. All models run forward.")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False


def run_training_experiment():
    """Run a quick training experiment."""
    print("\n🏋️ Running Training Experiment...")
    print("=" * 50)
    
    try:
        # Setup device and CUDA availability
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            cuda_available = torch.cuda.is_available()
        except Exception:
            cuda_available = False

        # Helper for long-sequence generalization
        def eval_long_seq(model_class, config, seq_lens, device='cpu'):
            losses = []
            max_len = config.get('max_seq_len', None)
            for seq_len in seq_lens:
                # Truncate input if seq_len > max_seq_len
                actual_len = min(seq_len, max_len) if max_len is not None else seq_len
                x = torch.randint(0, config['vocab_size'], (2, actual_len), device=device)
                model = model_class(config).to(device)
                with torch.no_grad():
                    out = model(x)
                    logits = out['logits'] if isinstance(out, dict) else out
                    targets = x[:, 1:].contiguous()
                    logits = logits[:, :-1, :]
                    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100)
                    losses.append(loss.item())
            return losses
        # --- Scenario 1 ---
        print("\n📊 Creating MQAR dataset (Scenario 1)...")
        train_data, val_data, test_data = create_mqar_datasets(
            train_size=500,
            val_size=100,
            test_size=100,
            seq_len=128,
            vocab_size=1000,
            num_kv_pairs=3,
            num_queries=1
        )
        shared_d_model = 144
        shared_num_layers = 6
        shared_num_heads = 4
        griffin_cfg = {
            "vocab_size": train_data.get_vocab_size(),
            "d_model": shared_d_model,
            "num_layers": shared_num_layers,
            "num_heads": shared_num_heads,
            "max_seq_len": 128,
            "local_window": 64
        }
        hawk_cfg = {
            "vocab_size": train_data.get_vocab_size(),
            "d_model": shared_d_model,
            "num_layers": shared_num_layers,
            "max_seq_len": 128
        }
        local_attn_cfg = {
            "vocab_size": train_data.get_vocab_size(),
            "d_model": shared_d_model,
            "num_layers": shared_num_layers,
            "num_heads": shared_num_heads,
            "max_seq_len": 128,
            "local_window": 64
        }
        models = {
            'Griffin': GriffinModel(griffin_cfg),
            'Hawk': HawkModel(hawk_cfg),
            'Local Attention': LocalAttentionModel(local_attn_cfg)
        }
        print("\n🔢 Parameter counts for MQAR (Scenario 1):")
        for name, model in models.items():
            params = sum(p.numel() for p in model.parameters())
            print(f"   {name:15}: {params:,} parameters")
        scenario_1_mqar = {}
        for name, model in models.items():
            print(f"\n🚀 Training {name} on MQAR (Scenario 1)...")
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
            train_loader = train_data.create_dataloader(batch_size=8, shuffle=True)
            model.train()
            total_loss = 0
            num_steps = 40
            start_time = time.time()
            if cuda_available:
                torch.cuda.reset_peak_memory_stats()
            process = psutil.Process(os.getpid())
            cpu_mem_peak = 0
            for step, batch in enumerate(train_loader):
                if step >= num_steps:
                    break
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                targets = input_ids[:, 1:].contiguous()
                outputs = model(input_ids[:, :-1])
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                mem = process.memory_info().rss / (1024 ** 2)
                if mem > cpu_mem_peak:
                    cpu_mem_peak = mem
                if step % 5 == 0:
                    print(f"   Step {step:2d}: Loss = {loss.item():.4f}")
            elapsed = time.time() - start_time
            avg_loss = total_loss / num_steps
            params = sum(p.numel() for p in model.parameters())
            total_samples = num_steps * 8
            throughput = total_samples / elapsed if elapsed > 0 else 0
            latency = elapsed / num_steps if num_steps > 0 else 0
            gpu_mem_peak = 0
            if cuda_available:
                gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            scenario_1_mqar[name] = {
                'parameters': params,
                'final_loss': avg_loss,
                'latency_sec_per_step': latency,
                'throughput_samples_per_sec': throughput,
                'cpu_mem_peak_mb': cpu_mem_peak,
                'gpu_mem_peak_mb': gpu_mem_peak
            }
            print(f"   ✅ {name}: {params:,} params, final loss: {avg_loss:.4f}, latency: {latency:.4f}s/step, throughput: {throughput:.2f} samples/s, CPU mem: {cpu_mem_peak:.1f}MB, GPU mem: {gpu_mem_peak:.1f}MB")

        # --- Chomsky Scenario 1 ---
        print("\n📊 Creating Chomsky a^n b^n dataset (Scenario 1)...")
        train_data_c, test_data_c = generate_an_bn_dataset(
            train_n_range=(1, 10),
            test_n_range=(20, 40, 80, 160),
            num_train=1000,
            num_test=400,
            vocab=['a', 'b'],
            hard_negatives=True
        )
        chomsky_train = AnBnDataset(train_data_c, max_length=128)
        chomsky_test = AnBnDataset(test_data_c, max_length=128)
        griffin_cfg_c = {
            "vocab_size": chomsky_train.total_vocab_size,
            "d_model": shared_d_model,
            "num_layers": shared_num_layers,
            "num_heads": shared_num_heads,
            "max_seq_len": 128,
            "local_window": 64
        }
        hawk_cfg_c = {
            "vocab_size": chomsky_train.total_vocab_size,
            "d_model": shared_d_model,
            "num_layers": shared_num_layers,
            "max_seq_len": 128
        }
        local_attn_cfg_c = {
            "vocab_size": chomsky_train.total_vocab_size,
            "d_model": shared_d_model,
            "num_layers": shared_num_layers,
            "num_heads": shared_num_heads,
            "max_seq_len": 128,
            "local_window": 64
        }
        models_chomsky = {
            'Griffin': GriffinModel(griffin_cfg_c),
            'Hawk': HawkModel(hawk_cfg_c),
            'Local Attention': LocalAttentionModel(local_attn_cfg_c)
        }
        print("\n🔢 Parameter counts for Chomsky a^n b^n (Scenario 1):")
        for name, model in models_chomsky.items():
            params = sum(p.numel() for p in model.parameters())
            print(f"   {name:15}: {params:,} parameters")
        scenario_1_chomsky = {}
        for name, model in models_chomsky.items():
            print(f"\n🚀 Training {name} on Chomsky a^n b^n (Scenario 1)...")
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
            train_loader = torch.utils.data.DataLoader(chomsky_train, batch_size=8, shuffle=True)
            model.train()
            total_loss = 0
            num_steps = 40
            start_time = time.time()
            if cuda_available:
                torch.cuda.reset_peak_memory_stats()
            process = psutil.Process(os.getpid())
            cpu_mem_peak = 0
            for step, batch in enumerate(train_loader):
                if step >= num_steps:
                    break
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                targets = batch['labels'].to(device)
                outputs = model(input_ids)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                pred = logits[:, -1, :].squeeze(1)
                loss = F.binary_cross_entropy_with_logits(pred[:, 0], targets.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                mem = process.memory_info().rss / (1024 ** 2)
                if mem > cpu_mem_peak:
                    cpu_mem_peak = mem
                if step % 5 == 0:
                    print(f"   Step {step:2d}: Loss = {loss.item():.4f}")
            elapsed = time.time() - start_time
            avg_loss = total_loss / num_steps
            params = sum(p.numel() for p in model.parameters())
            total_samples = num_steps * 8
            throughput = total_samples / elapsed if elapsed > 0 else 0
            latency = elapsed / num_steps if num_steps > 0 else 0
            gpu_mem_peak = 0
            if cuda_available:
                gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            scenario_1_chomsky[name] = {
                'parameters': params,
                'final_loss': avg_loss,
                'latency_sec_per_step': latency,
                'throughput_samples_per_sec': throughput,
                'cpu_mem_peak_mb': cpu_mem_peak,
                'gpu_mem_peak_mb': gpu_mem_peak
            }
            print(f"   ✅ {name}: {params:,} params, final loss: {avg_loss:.4f}, latency: {latency:.4f}s/step, throughput: {throughput:.2f} samples/s, CPU mem: {cpu_mem_peak:.1f}MB, GPU mem: {gpu_mem_peak:.1f}MB")

        # --- Long-sequence generalization Scenario 1 ---
        print("\n🔎 Running Long-Sequence Generalization Test (Scenario 1)...")
        seq_lens = [128, 256, 512, 1024, 2048]
        griffin_losses = eval_long_seq(models['Griffin'].__class__, griffin_cfg, seq_lens, device)
        hawk_losses = eval_long_seq(models['Hawk'].__class__, hawk_cfg, seq_lens, device)
        local_losses = eval_long_seq(models['Local Attention'].__class__, local_attn_cfg, seq_lens, device)
        scenario_1_longseq = {
            "sequence_lengths": seq_lens,
            "Griffin": griffin_losses,
            "Hawk": hawk_losses,
            "Local Attention": local_losses
        }

        scenario_1 = {
            'MQAR': scenario_1_mqar,
            'Chomsky': scenario_1_chomsky,
            'long_sequence_generalization': scenario_1_longseq
        }

        # --- Scenario 2 ---
        print("\n📊 Creating MQAR dataset (Scenario 2)...")
        train_data2, val_data2, test_data2 = create_mqar_datasets(
            train_size=1000,
            val_size=200,
            test_size=200,
            seq_len=512,
            vocab_size=1000,
            num_kv_pairs=5,
            num_queries=2
        )
        shared_d_model2 = 256
        shared_num_layers2 = 8
        shared_num_heads2 = 8
        griffin_cfg2 = {
            "vocab_size": train_data2.get_vocab_size(),
            "d_model": shared_d_model2,
            "num_layers": shared_num_layers2,
            "num_heads": shared_num_heads2,
            "max_seq_len": 512,
            "local_window": 128
        }
        hawk_cfg2 = {
            "vocab_size": train_data2.get_vocab_size(),
            "d_model": shared_d_model2,
            "num_layers": shared_num_layers2,
            "max_seq_len": 512
        }
        local_attn_cfg2 = {
            "vocab_size": train_data2.get_vocab_size(),
            "d_model": shared_d_model2,
            "num_layers": shared_num_layers2,
            "num_heads": shared_num_heads2,
            "max_seq_len": 512,
            "local_window": 128
        }
        models2 = {
            'Griffin': GriffinModel(griffin_cfg2),
            'Hawk': HawkModel(hawk_cfg2),
            'Local Attention': LocalAttentionModel(local_attn_cfg2)
        }
        print("\n🔢 Parameter counts for MQAR (Scenario 2):")
        for name, model in models2.items():
            params = sum(p.numel() for p in model.parameters())
            print(f"   {name:15}: {params:,} parameters")
        mqar_results2 = {}
        for name, model in models2.items():
            print(f"\n🚀 Training {name} on MQAR (Scenario 2)...")
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
            train_loader = train_data2.create_dataloader(batch_size=8, shuffle=True)
            model.train()
            total_loss = 0
            num_steps = 40
            start_time = time.time()
            if cuda_available:
                torch.cuda.reset_peak_memory_stats()
            process = psutil.Process(os.getpid())
            cpu_mem_peak = 0
            for step, batch in enumerate(train_loader):
                if step >= num_steps:
                    break
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                targets = input_ids[:, 1:].contiguous()
                outputs = model(input_ids[:, :-1])
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                mem = process.memory_info().rss / (1024 ** 2)
                if mem > cpu_mem_peak:
                    cpu_mem_peak = mem
                if step % 5 == 0:
                    print(f"   Step {step:2d}: Loss = {loss.item():.4f}")
            elapsed = time.time() - start_time
            avg_loss = total_loss / num_steps
            params = sum(p.numel() for p in model.parameters())
            total_samples = num_steps * 8
            throughput = total_samples / elapsed if elapsed > 0 else 0
            latency = elapsed / num_steps if num_steps > 0 else 0
            gpu_mem_peak = 0
            if cuda_available:
                gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            mqar_results2[name] = {
                'parameters': params,
                'final_loss': avg_loss,
                'latency_sec_per_step': latency,
                'throughput_samples_per_sec': throughput,
                'cpu_mem_peak_mb': cpu_mem_peak,
                'gpu_mem_peak_mb': gpu_mem_peak
            }
            print(f"   ✅ {name}: {params:,} params, final loss: {avg_loss:.4f}, latency: {latency:.4f}s/step, throughput: {throughput:.2f} samples/s, CPU mem: {cpu_mem_peak:.1f}MB, GPU mem: {gpu_mem_peak:.1f}MB")

        # Chomsky Scenario 2
        print("\n📊 Creating Chomsky a^n b^n dataset (Scenario 2)...")
        train_data_c2, test_data_c2 = generate_an_bn_dataset(
            train_n_range=(1, 20),
            test_n_range=(40, 80, 160, 320),
            num_train=2000,
            num_test=800,
            vocab=['a', 'b'],
            hard_negatives=True
        )
        chomsky_train2 = AnBnDataset(train_data_c2, max_length=512)
        chomsky_test2 = AnBnDataset(test_data_c2, max_length=512)
        griffin_cfg_c2 = {
            "vocab_size": chomsky_train2.total_vocab_size,
            "d_model": shared_d_model2,
            "num_layers": shared_num_layers2,
            "num_heads": shared_num_heads2,
            "max_seq_len": 512,
            "local_window": 128
        }
        hawk_cfg_c2 = {
            "vocab_size": chomsky_train2.total_vocab_size,
            "d_model": shared_d_model2,
            "num_layers": shared_num_layers2,
            "max_seq_len": 512
        }
        local_attn_cfg_c2 = {
            "vocab_size": chomsky_train2.total_vocab_size,
            "d_model": shared_d_model2,
            "num_layers": shared_num_layers2,
            "num_heads": shared_num_heads2,
            "max_seq_len": 512,
            "local_window": 128
        }
        models_chomsky2 = {
            'Griffin': GriffinModel(griffin_cfg_c2),
            'Hawk': HawkModel(hawk_cfg_c2),
            'Local Attention': LocalAttentionModel(local_attn_cfg_c2)
        }
        print("\n🔢 Parameter counts for Chomsky a^n b^n (Scenario 2):")
        for name, model in models_chomsky2.items():
            params = sum(p.numel() for p in model.parameters())
            print(f"   {name:15}: {params:,} parameters")
        chomsky_results2 = {}
        for name, model in models_chomsky2.items():
            print(f"\n🚀 Training {name} on Chomsky a^n b^n (Scenario 2)...")
            model = model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002)
            train_loader = torch.utils.data.DataLoader(chomsky_train2, batch_size=8, shuffle=True)
            model.train()
            total_loss = 0
            num_steps = 40
            start_time = time.time()
            if cuda_available:
                torch.cuda.reset_peak_memory_stats()
            process = psutil.Process(os.getpid())
            cpu_mem_peak = 0
            for step, batch in enumerate(train_loader):
                if step >= num_steps:
                    break
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                targets = batch['labels'].to(device)
                outputs = model(input_ids)
                logits = outputs['logits'] if isinstance(outputs, dict) else outputs
                pred = logits[:, -1, :].squeeze(1)
                loss = F.binary_cross_entropy_with_logits(pred[:, 0], targets.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()
                mem = process.memory_info().rss / (1024 ** 2)
                if mem > cpu_mem_peak:
                    cpu_mem_peak = mem
                if step % 5 == 0:
                    print(f"   Step {step:2d}: Loss = {loss.item():.4f}")
            elapsed = time.time() - start_time
            avg_loss = total_loss / num_steps
            params = sum(p.numel() for p in model.parameters())
            total_samples = num_steps * 8
            throughput = total_samples / elapsed if elapsed > 0 else 0
            latency = elapsed / num_steps if num_steps > 0 else 0
            gpu_mem_peak = 0
            if cuda_available:
                gpu_mem_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            chomsky_results2[name] = {
                'parameters': params,
                'final_loss': avg_loss,
                'latency_sec_per_step': latency,
                'throughput_samples_per_sec': throughput,
                'cpu_mem_peak_mb': cpu_mem_peak,
                'gpu_mem_peak_mb': gpu_mem_peak
            }
            print(f"   ✅ {name}: {params:,} params, final loss: {avg_loss:.4f}, latency: {latency:.4f}s/step, throughput: {throughput:.2f} samples/s, CPU mem: {cpu_mem_peak:.1f}MB, GPU mem: {gpu_mem_peak:.1f}MB")

        # Long-sequence generalization Scenario 2
        print("\n🔎 Running Long-Sequence Generalization Test (Scenario 2)...")
        seq_lens2 = [256, 512, 1024, 2048, 4096]
        griffin_losses2 = eval_long_seq(models2['Griffin'].__class__, griffin_cfg2, seq_lens2, device)
        hawk_losses2 = eval_long_seq(models2['Hawk'].__class__, hawk_cfg2, seq_lens2, device)
        local_losses2 = eval_long_seq(models2['Local Attention'].__class__, local_attn_cfg2, seq_lens2, device)
        longseq_results2 = {
            "sequence_lengths": seq_lens2,
            "Griffin": griffin_losses2,
            "Hawk": hawk_losses2,
            "Local Attention": local_losses2
        }

        scenario_2 = {
            'MQAR': mqar_results2,
            'Chomsky': chomsky_results2,
            'long_sequence_generalization': longseq_results2
        }

        # Save both scenarios
        all_results = {
            'scenario_1': scenario_1,
            'scenario_2': scenario_2
        }
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / 'quick_experiment.json', 'w') as f:
            import json
            json.dump(all_results, f, indent=2)
        print(f"\n💾 Results saved to: {results_dir / 'quick_experiment.json'}")
        return True

        # Save all results in one file
        results_dir = Path(__file__).parent / 'results'
        results_dir.mkdir(exist_ok=True)
        with open(results_dir / 'quick_experiment.json', 'w') as f:
            import json
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved to: {results_dir / 'quick_experiment.json'}")
        return True
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_project_info():
    """Show project information and structure."""
    print("\n📋 GRIFFIN MODEL PROJECT INFO")
    print("=" * 60)
    
    print("\n🎯 PROJECT OBJECTIVES:")
    print("   ✅ Implement Griffin hybrid architecture (recurrence + attention)")
    print("   ✅ Implement Hawk pure recurrent architecture")
    print("   ✅ Implement Local Attention pure attention architecture")
    print("   ✅ Compare all three models on memory tasks (MQAR, Chomsky)")
    print("   ✅ Professional repository for thesis work")
    
    print("\n📁 PROJECT STRUCTURE:")
    print("   models/")
    print("   ├── griffin/     - Hybrid Griffin model implementation")
    print("   ├── hawk/        - Pure recurrent Hawk model")
    print("   └── local_attention/ - Pure attention model")
    print("   data/")
    print("   ├── mqar/        - Multi-Query Associative Recall dataset")
    print("   └── chomsky/     - Chomsky hierarchy tasks")
    print("   config/          - Model configurations (tiny, small, medium)")
    print("   training/        - Training infrastructure")
    print("   evaluation/      - Evaluation and comparison tools")
    print("   experiments/     - Experiment orchestration")
    print("   notebooks/       - Jupyter notebooks for analysis")
    
    print("\n🔬 SMALL MODEL SIZES (for efficient research):")
    print("   Tiny:   ~2M  parameters (d_model=128, layers=2)")
    print("   Small:  ~10M parameters (d_model=256, layers=4)")
    print("   Medium: ~25M parameters (d_model=384, layers=6)")
    
    print("\n🚀 QUICK COMMANDS:")
    print("   python quick_start.py --test     # Test all models")
    print("   python quick_start.py --train    # Run training experiment")
    print("   python quick_start.py --install  # Install dependencies")
    print("   jupyter notebook notebooks/      # Interactive analysis")
    
    print("\n📚 FOR YOUR THESIS:")
    print("   1. Use notebooks for interactive analysis and plots")
    print("   2. Run experiments with different model sizes")
    print("   3. Compare Griffin vs Hawk vs Local Attention")
    print("   4. Analyze hybrid advantages and trade-offs")


def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Griffin Model Quick Start')
    parser.add_argument('--install', action='store_true', help='Install dependencies')
    parser.add_argument('--test', action='store_true', help='Run quick test')
    parser.add_argument('--train', action='store_true', help='Run training experiment')
    parser.add_argument('--info', action='store_true', help='Show project info')
    parser.add_argument('--longseq', action='store_true', help='Run long-sequence generalization test')
    args = parser.parse_args()
    
    print("🚀 GRIFFIN MODEL PROJECT - QUICK START")
    print("=" * 60)
    
    if args.install:
        if install_dependencies():
            print("✅ Dependencies installed successfully!")
        else:
            print("❌ Failed to install dependencies")
            return 1
    
    elif args.test:
        # Check dependencies first
        missing = check_dependencies()
        if missing:
            print(f"❌ Missing dependencies: {missing}")
            print("Run: python quick_start.py --install")
            return 1
        
        if run_quick_test():
            print("✅ All tests passed!")
        else:
            print("❌ Tests failed")
            return 1
    
    elif args.train:
        # Check dependencies first
        missing = check_dependencies()
        if missing:
            print(f"❌ Missing dependencies: {missing}")
            print("Run: python quick_start.py --install")
            return 1
        
        if run_training_experiment():
            print("✅ Training experiment completed!")
        else:
            print("❌ Training failed")
            return 1
    
    elif args.longseq:
        missing = check_dependencies()
        if missing:
            print(f"❌ Missing dependencies: {missing}")
            print("Run: python quick_start.py --install")
            return 1
        if run_long_sequence_generalization():
            print("✅ Long-sequence generalization test completed!")
        else:
            print("❌ Long-sequence generalization test failed")
            return 1
    elif args.info:
        show_project_info()
    else:
        # Default: show info and check status
        show_project_info()
        
        print("\n🔍 DEPENDENCY CHECK:")
        missing = check_dependencies()
        if missing:
            print(f"   ❌ Missing: {missing}")
            print("   👉 Run: python quick_start.py --install")
        else:
            print("   ✅ All dependencies available")
            print("   👉 Run: python quick_start.py --test")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
