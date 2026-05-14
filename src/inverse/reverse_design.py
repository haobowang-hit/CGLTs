import os 
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import optuna
import math
from tqdm import tqdm
import argparse



# Import enhanced model

from models.vae import (
    EnhancedConditionalEncoder,
    EnhancedConditionalDecoder, 
    EnhancedANNMapper,
    dtw_2d_loss
)

from utils.dataloader import load_normalization_params
from utils.utils import load_model



def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced Inverse Design with Bayesian Optimization")

    

    # Path configuration
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints',

                       help='Directory containing trained models')

    parser.add_argument('--test_curve_csv', type=str, required=True,

                       help='CSV file containing independent test curves')

    parser.add_argument('--test_curve_dir', type=str, required=True,

                       help='Directory containing test curve files')

    parser.add_argument('--save_dir', type=str, default='./inverse_results',

                       help='Directory to save inverse design results')

    

    # Model parameters

    parser.add_argument('--z_dim', type=int, default=16)

    parser.add_argument('--feature_dim', type=int, default=8)

    parser.add_argument('--device', type=str, default='cuda')

    

    # Optimization parameters

    parser.add_argument('--n_trials', type=int, default=100,

                       help='Number of Bayesian optimization trials')

    parser.add_argument('--max_samples', type=int, default=500,

                       help='Maximum number of test samples to process')

    

    return parser.parse_args()



def createside(H, Angle, Radius, Lumbus):

    """Geometric constraint verification function

    Input parameters: H, Angle, Radius, Lumbus (format during neural network training)"""

    sr = H / 2.0

    r1 = Radius

    a1 = Angle

    w = Lumbus  # Lumbus parameters are used here

    try:

        r2 = (sr - r1 + r1 * math.cos(math.radians(a1))) / (1 - math.cos(math.radians(a1)))

        Design_space = 120.0

        w1 = Design_space * 0.5 - a1 * r1 * math.pi / 180 - a1 * r2 * math.pi / 90 - w * 0.5

        return r2, w1

    except (ZeroDivisionError, ValueError):

        return -1, -1  # Return invalid value



def validate_design_constraints(H, Angle, Radius, Lumbus):

    """Verify that design parameters meet constraints

    Input parameter format: H, Angle, Radius, Lumbus (format during neural network training)

    Parameter range: H∈[24,36]mm (even number), α∈[30°,80°], r∈[10,H/2]mm, L∈[1,10]mm"""

    # basic range constraints

    if not (24 <= H <= 36):  # H ∈ [24,36] mm

        return False

    if H % 2 != 0:  # Make sure H is an even number

        return False

    if not (30 <= Angle <= 80):  # α ∈ [30°,80°]

        return False

    if not (10 <= Radius <= H//2):  # r ∈ [10, H/2] mm

        return False

    if not (1 <= Lumbus <= 10):  # L ∈ [1,10] mm

        return False

    

    # Geometric constraints (createside function uses H, Angle, Radius, Lumbus)

    r2, w1 = createside(H, Angle, Radius, Lumbus)

    if r2 <= 2 or w1 <= 2:

        return False

    

    return True



class EnhancedInverseDesigner:

    def __init__(self, encoder, decoder, mapper, normalization_params, device='cuda'):

        self.encoder = encoder.eval()

        self.decoder = decoder.eval()

        self.mapper = mapper.eval()

        self.device = device

        self.norm_params = normalization_params

        

        # Set parameter range (based on the definition in the paper, H is limited to even numbers)

        self.param_ranges = {

            'H': (24, 36),           # H ∈ [24,36] mm, even number

            'Lumbus': (1, 10),       # L ∈ [1,10] mm  

            'Angle': (30, 80),       # α ∈ [30°,80°] with 5° increments

            'Radius': (10, 18),      # r ∈ [10,H/2] mm, when H is maximum 36, H/2=18

        }

    

    def normalize_parameters(self, params_raw):

        """Normalized design parameters

        Input params_raw format: [H1, Lumbus1, Angle1, Radius1, H2, Lumbus2, Angle2, Radius2]"""

        if self.norm_params is None:

            normalized = []

            # Corresponding parameter range during neural network training [H, Lumbus, Angle, Radius] * 2

            ranges = [

                (24, 36), (1, 10), (30, 80), (10, 18),  # Section 1: H1, Lumbus1, Angle1, Radius1

                (24, 36), (1, 10), (30, 80), (10, 18)   # Section 2: H2, Lumbus2, Angle2, Radius2

            ]

            

            for param, (min_val, max_val) in zip(params_raw, ranges):

                normalized.append((param - min_val) / (max_val - min_val))

            

            return np.array(normalized, dtype=np.float32)

        else:

            # Use saved normalization parameters

            scaler = self.norm_params[0]

            return scaler.transform([params_raw])[0]

    

    def predict_curve_from_params(self, params_raw):

        """Predicting response curves from design parameters"""

        # Normalization parameters

        params_norm = self.normalize_parameters(params_raw)

        params_tensor = torch.tensor(params_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

        

        with torch.no_grad():

            # Predict latent vectors via mapper

            pred_z = self.mapper(params_tensor)

            # Predict the curve through the decoder

            pred_curve = self.decoder(pred_z, params_tensor)

            

        return pred_curve[0].cpu().numpy()

    

    def denormalize_curve(self, curve_norm):

        """denormalized curve"""

        if self.norm_params is None:

            return curve_norm

        

        # Use saved normalization parameters

        _, curve_min, curve_max = self.norm_params

        return curve_norm * (curve_max - curve_min) + curve_min

    

    def optimize_single_curve_bayes(self, target_curve_np, n_trials=100):

        """Single-curve inverse design using Bayesian optimization"""

        

        # Prepare target curve

        target_curve = torch.tensor(target_curve_np, dtype=torch.float32, device=self.device)

        

        # If there are normalization parameters, normalize the target curve first

        if self.norm_params is not None:

            _, curve_min, curve_max = self.norm_params

            curve_min = torch.tensor(curve_min, dtype=torch.float32, device=self.device)

            curve_max = torch.tensor(curve_max, dtype=torch.float32, device=self.device)

            target_curve = (target_curve - curve_min) / (curve_max - curve_min + 1e-8)

        

        target_curve = target_curve.unsqueeze(0)

        

        def objective(trial):

            """Optuna objective function"""

            def sample_design_section(idx):

                """Sample design parameters for a single section

                Return format: [H, Lumbus, Angle, Radius] (format during neural network training)

                Parameter range: H∈[24,36]mm (even number), α∈[30°,80°], r∈[10,H/2]mm, L∈[1,10]mm"""

                # H must be an even number, an even number in the range [24,36]: 24, 26, 28, 30, 32, 34, 36

                even_H_values = list(range(24, 37, 2))  # [24, 26, 28, 30, 32, 34, 36]

                H = trial.suggest_categorical(f'H{idx}', even_H_values)

                

                Lumbus = trial.suggest_int(f'Lumbus{idx}', 1, 10)  # L ∈ [1,10] mm

                Angle = trial.suggest_categorical(f'Angle{idx}', list(range(30, 85, 5)))  # α ∈ [30°,80°] with 5° increments

                Radius = trial.suggest_int(f'Radius{idx}', 10, min(H//2, 18))  # r ∈ [10, H/2] mm

                

                # Verify geometric constraints (using H, Angle, Radius, Lumbus)

                if not validate_design_constraints(H, Angle, Radius, Lumbus):

                    raise optuna.exceptions.TrialPruned()

                

                return [H, Lumbus, Angle, Radius]  # Neural network training format

            

            try:

                # Sample parameters of two sections

                section1 = sample_design_section(1)

                section2 = sample_design_section(2)

                params_raw = section1 + section2

                

            except optuna.exceptions.TrialPruned:

                return float('inf')

            

            # Normalization parameters

            params_norm = self.normalize_parameters(params_raw)

            params_tensor = torch.tensor(params_norm, dtype=torch.float32).unsqueeze(0).to(self.device)

            

            with torch.no_grad():

                # prediction curve

                pred_z = self.mapper(params_tensor)

                pred_curve = self.decoder(pred_z, params_tensor)

                

                # If denormalization is needed for comparison

                if self.norm_params is not None:

                    pred_denorm = pred_curve * (curve_max - curve_min + 1e-8) + curve_min

                    target_denorm = target_curve * (curve_max - curve_min + 1e-8) + curve_min

                    

                    # Y-axis scaling (according to your original code)

                    pred_denorm[..., 1] = pred_denorm[..., 1] / 1000.0

                    target_denorm[..., 1] = target_denorm[..., 1] / 1000.0

                    

                    # Calculate DTW loss

                    loss = dtw_2d_loss(pred_denorm, target_denorm)

                else:

                    # Calculate the loss directly in the normalized space

                    loss = dtw_2d_loss(pred_curve, target_curve)

            

            return loss.item()

        

        # Create an optimization study

        study = optuna.create_study(direction="minimize")

        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        

        try:

            # Extract the best parameters (neural network training format: H, Lumbus, Angle, Radius)

            best_params_nn_format = [

                study.best_params['H1'], study.best_params['Lumbus1'],

                study.best_params['Angle1'], study.best_params['Radius1'],

                study.best_params['H2'], study.best_params['Lumbus2'],

                study.best_params['Angle2'], study.best_params['Radius2']

            ]

            best_loss = study.best_value

            

            return best_params_nn_format, best_loss, study

        except Exception as e:

            print(f"Failed to extract best parameters: {e}")

            return None, None, study



def load_test_samples(test_csv_file):

    """Load independent test samples"""

    try:

        df = pd.read_csv(test_csv_file)

        print(f"Loaded {len(df)} test samples from {test_csv_file}")

        return df

    except Exception as e:

        print(f"Error loading test samples: {e}")

        return None



def load_target_curve(curve_dir, jobnum):

    """Load target curve"""

    curve_file = os.path.join(curve_dir, f"cgltNonLinear_no{jobnum}_resampled.csv")

    if os.path.exists(curve_file):

        return pd.read_csv(curve_file, header=None).values.astype(np.float32)

    else:

        return None



def main():

    parser = argparse.ArgumentParser(description="Enhanced Inverse Design with Bayesian Optimization")

    

    parser.add_argument('--checkpoint_dir', type=str,
                       default='./checkpoints',

                       help='Directory containing trained models')

    parser.add_argument('--test_curve_csv', type=str,
                       default='./data/input/independent_test_pairs.csv',

                       help='CSV file containing independent test curves')

    parser.add_argument('--test_curve_dir', type=str,
                       default='./data/output',

                       help='Directory containing test curve files')

    parser.add_argument('--save_dir', type=str,
                       default='./inverse_results',

                       help='Directory to save inverse design results')

    

    # Model parameters

    parser.add_argument('--z_dim', type=int, default=16)

    parser.add_argument('--feature_dim', type=int, default=8)

    parser.add_argument('--device', type=str, default='cuda')

    

    # Optimization parameters

    parser.add_argument('--n_trials', type=int, default=100,

                       help='Number of Bayesian optimization trials')

    parser.add_argument('--max_samples', type=int, default=500,

                       help='Maximum number of test samples to process')

    parser.add_argument('--start_job', type=int, default=10001,

                       help='Starting job number for test set')

    parser.add_argument('--end_job', type=int, default=10500,

                       help='Ending job number for test set')

    

    args = parser.parse_args()

    

    # Create save directory

    os.makedirs(args.save_dir, exist_ok=True)

    

    print("Loading Enhanced CVAE-Attention models...")

    

    # Load model

    encoder = EnhancedConditionalEncoder(z_dim=args.z_dim, feature_dim=args.feature_dim).to(args.device)

    decoder = EnhancedConditionalDecoder(z_dim=args.z_dim, feature_dim=args.feature_dim).to(args.device)

    mapper = EnhancedANNMapper(feature_dim=args.feature_dim, z_dim=args.z_dim).to(args.device)

    

    # Load weights

    model_files = [

        ('best_encoder.pt', 'best_decoder.pt', 'best_mapper.pt'),

        ('final_encoder.pt', 'final_decoder.pt', 'final_mapper.pt'),

        ('encoder.pt', 'decoder.pt', 'mapper.pt')

    ]

    

    loaded = False

    for enc_file, dec_file, map_file in model_files:

        try:

            encoder_path = f"{args.checkpoint_dir}/{enc_file}"

            decoder_path = f"{args.checkpoint_dir}/{dec_file}"

            mapper_path = f"{args.checkpoint_dir}/{map_file}"

            

            if all(os.path.exists(p) for p in [encoder_path, decoder_path, mapper_path]):

                load_model(encoder, encoder_path, args.device)

                load_model(decoder, decoder_path, args.device)

                load_model(mapper, mapper_path, args.device)

                print(f"Loaded models: {enc_file}, {dec_file}, {map_file}")

                loaded = True

                break

        except Exception as e:

            print(f"Failed to load {enc_file}, {dec_file}, {map_file}: {e}")

            continue

    

    if not loaded:

        print("Error: Could not load any model files!")

        return

    

    # Load normalization parameters

    normalization_params = None

    try:

        normalization_params = load_normalization_params(args.checkpoint_dir)

        print("Loaded normalization parameters")

    except Exception as e:

        print(f"Warning: Could not load normalization parameters: {e}")

    

    # Create a reverse designer

    inverse_designer = EnhancedInverseDesigner(

        encoder, decoder, mapper, normalization_params, args.device

    )

    

    # Load test samples

    test_samples_df = load_test_samples(args.test_curve_csv)

    if test_samples_df is None:

        return

    

    print(f"Loaded test samples: {len(test_samples_df)} samples")

    print(f"Job number range: {test_samples_df['jobnum'].min()} - {test_samples_df['jobnum'].max()}")

    

    # Filter samples within a specified range

    test_samples_df = test_samples_df[

        (test_samples_df['jobnum'] >= args.start_job) & 

        (test_samples_df['jobnum'] <= args.end_job)

    ]

    print(f"Filtered samples in range [{args.start_job}, {args.end_job}]: {len(test_samples_df)} samples")

    

    # Limit the number of samples processed

    if len(test_samples_df) > args.max_samples:

        test_samples_df = test_samples_df.head(args.max_samples)

        print(f"Limited to {args.max_samples} samples")

    

    # Check whether the test curve file exists

    available_curves = []

    missing_curves = []

    for _, row in test_samples_df.iterrows():

        jobnum = int(row['jobnum'])

        curve_file = os.path.join(args.test_curve_dir, f"cgltNonLinear_no{jobnum}_resampled.csv")

        if os.path.exists(curve_file):

            available_curves.append(jobnum)

        else:

            missing_curves.append(jobnum)

    

    print(f"Available curve files: {len(available_curves)}")

    print(f"Missing curve files: {len(missing_curves)}")

    if missing_curves and len(missing_curves) <= 10:

        print(f"Missing jobs: {missing_curves}")

    

    # Only processes samples with curve files

    test_samples_df = test_samples_df[test_samples_df['jobnum'].isin(available_curves)]

    print(f"Final samples to process: {len(test_samples_df)}")

    

    if len(test_samples_df) == 0:

        print("No valid samples to process!")

        return

    

    # Perform reverse engineering

    print(f"Starting enhanced inverse design for {len(test_samples_df)} samples...")

    

    optimized_params = []

    losses = []

    failed_samples = []

    

    for idx, row in tqdm(test_samples_df.iterrows(), total=len(test_samples_df), desc="Inverse Design"):

        jobnum = int(row['jobnum'])

        

        # Load target curve

        target_curve = load_target_curve(args.test_curve_dir, jobnum)

        if target_curve is None:

            print(f"Missing target curve for job {jobnum}")

            failed_samples.append(jobnum)

            continue

        

        # Perform reverse engineering

        try:

            best_params, best_loss, study = inverse_designer.optimize_single_curve_bayes(

                target_curve, n_trials=args.n_trials

            )

            

            if best_params is None:

                print(f"[{jobnum}] ❌ No feasible solution found.")

                optimized_params.append([jobnum] + [np.nan]*8)

                losses.append([jobnum, np.nan])

                failed_samples.append(jobnum)

            else:

                # Convert parameter format: from neural network format [H, Lumbus, Angle, Radius]

                # Directly output the original format [H, Lumbus, angle, radius]

                output_params = [

                    best_params[0],     # H1 = H1 (directly output H value)

                    best_params[1],     # Lumbus1 = Lumbus1

                    best_params[2],     # angle1 = Angle1

                    best_params[3],     # radius1 = Radius1

                    best_params[4],     # H2 = H2 (directly output H value)

                    best_params[5],     # Lumbus2 = Lumbus2

                    best_params[6],     # angle2 = Angle2

                    best_params[7],     # radius2 = Radius2

                ]

                

                optimized_params.append([jobnum] + output_params)

                losses.append([jobnum, best_loss])

                

                # Verify the plausibility of the results (validate using neural network format)

                # best_params format: [H1, Lumbus1, Angle1, Radius1, H2, Lumbus2, Angle2, Radius2]

                all_valid = all(validate_design_constraints(

                    best_params[i*4], best_params[i*4+2], best_params[i*4+3], best_params[i*4+1]

                ) for i in range(2))

                

                if not all_valid:

                    print(f"Warning: Job {jobnum} produced invalid design parameters")

                

                # Verify that H is indeed an even number

                H1, H2 = output_params[0], output_params[4]

                if H1 % 2 != 0 or H2 % 2 != 0:

                    print(f"Warning: H values are not even - H1: {H1}, H2: {H2}")

        

        except Exception as e:

            print(f"Error processing job {jobnum}: {e}")

            failed_samples.append(jobnum)

            continue

    

    # Save results

    print("Saving results...")

    

    opt_df = pd.DataFrame(optimized_params, columns=[

        "jobnum", "H1", "Lumbus1", "angle1", "radius1", 

        "H2", "Lumbus2", "angle2", "radius2"

    ])

    opt_df.to_csv(os.path.join(args.save_dir, "enhanced_inverse_design_results.csv"), index=False)

    

    # loss result

    loss_df = pd.DataFrame(losses, columns=["jobnum", "Loss"])

    loss_df.to_csv(os.path.join(args.save_dir, "enhanced_inverse_losses.csv"), index=False)

    

    # failed sample

    if failed_samples:

        fail_df = pd.DataFrame(failed_samples, columns=["jobnum"])

        fail_df.to_csv(os.path.join(args.save_dir, "failed_inverse_samples.csv"), index=False)

    

    # Generate statistical reports

    successful_samples = len(optimized_params) - len(failed_samples)

    success_rate = successful_samples / len(test_samples_df) * 100 if len(test_samples_df) > 0 else 0

    

    print(f"\n✅ Enhanced Inverse Design completed!")

    print(f"Total samples processed: {len(test_samples_df)}")

    print(f"Successful inversions: {successful_samples}")

    print(f"Failed samples: {len(failed_samples)}")

    print(f"Success rate: {success_rate:.2f}%")

    

    if successful_samples > 0:

        valid_losses = [loss for _, loss in losses if not np.isnan(loss)]

        if valid_losses:

            print(f"Average loss: {np.mean(valid_losses):.6f} ± {np.std(valid_losses):.6f}")

            print(f"Best loss: {np.min(valid_losses):.6f}")

    

    print(f"Results saved to: {args.save_dir}")

    

    # Verify that the generated H values ​​are all even numbers

    if successful_samples > 0:

        non_even_count = 0

        for params in optimized_params:

            if len(params) > 8:  # Make sure you have enough parameters

                H1, H2 = params[1], params[5]  # H1 and H2

                if H1 % 2 != 0 or H2 % 2 != 0:

                    non_even_count += 1

        

        print(f"Non-even H values count: {non_even_count}")

        if non_even_count == 0:

            print("✅ All H values are even numbers!")

    

    # Generate detailed reports

    report_content = f"""

Enhanced Inverse Design Report (Even H Values)

=============================================



Dataset Information:

- Test CSV: {args.test_curve_csv}

- Curve Directory: {args.test_curve_dir}

- Job Range: {args.start_job} - {args.end_job}



Processing Summary:

- Total test samples: {len(test_samples_df)}

- Available curves: {len(available_curves)}

- Missing curves: {len(missing_curves)}

- Successful inversions: {successful_samples}

- Failed samples: {len(failed_samples)}

- Success rate: {success_rate:.2f}%



Optimization Settings:

- Trials per sample: {args.n_trials}

- Z dimension: {args.z_dim}

- Device: {args.device}

- H constraint: Even values only (24, 26, 28, 30, 32, 34, 36)



Output Format:

- Direct H values (not Sectionalradius)

- Parameters: [H1, Lumbus1, angle1, radius1, H2, Lumbus2, angle2, radius2]



Results saved to: {args.save_dir}



Note: All H values are constrained to be even numbers.

Output contains original H values rather than derived Sectionalradius values.

"""

    

    with open(os.path.join(args.save_dir, "inverse_design_report.txt"), 'w') as f:

        f.write(report_content)

    

    print("Detailed report saved to inverse_design_report.txt")



if __name__ == "__main__":

    main()
