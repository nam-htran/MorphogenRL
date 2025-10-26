import tensorflow as tf
import torch
import numpy as np
import os

# Suppress unnecessary TensorFlow 1.x warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
if tf.__version__.startswith('1.'):
    tf.logging.set_verbosity(tf.logging.ERROR)

def convert_tf1_to_pytorch(project_root=None):
    """
    Convert TensorFlow 1.x checkpoint weights to a PyTorch state_dict.
    """
    if project_root is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    TF_WEIGHTS_DIR = os.path.join(project_root, 'TeachMyAgent', 'environments', 'envs', 'PCGAgents', 'CPPN', 'weights', 'same_ground_ceiling_cppn')
    PYTORCH_WEIGHTS_PATH = os.path.join(project_root, 'TeachMyAgent', 'environments', 'envs', 'PCGAgents', 'CPPN', 'weights', 'same_ground_ceiling_cppn_pytorch.pt')

    # Skip conversion if PyTorch weights already exist
    if os.path.exists(PYTORCH_WEIGHTS_PATH):
        print(f"PyTorch weights already exist at: {PYTORCH_WEIGHTS_PATH}. Skipping conversion.")
        return
    
    TF_VAR_TO_PYTORCH_NAME = {
        'Variable':   '0.weight', 
        'Variable_1': '2.weight', 
        'Variable_2': '4.weight',
        'Variable_3': '6.weight', 
        'Variable_4': '8.weight',
    }

    print("Starting weight conversion...")
    checkpoint_path = tf.train.latest_checkpoint(TF_WEIGHTS_DIR)
    if not checkpoint_path:
        print(f"ERROR: No checkpoint found in {TF_WEIGHTS_DIR}.")
        return

    try:
        reader = tf.train.NewCheckpointReader(checkpoint_path)
        pytorch_state_dict = {}
        
        print("\n--- Variables found in checkpoint ---")
        for key in sorted(reader.get_variable_to_shape_map().keys()):
            print(f"  Name: {key}")
        print("------------------------------------------")
        
        for tf_name, pt_name in TF_VAR_TO_PYTORCH_NAME.items():
            if reader.has_tensor(tf_name):
                weight_numpy = reader.get_tensor(tf_name)
                # Transpose to match PyTorch Linear (out_features, in_features)
                pytorch_state_dict[pt_name] = torch.from_numpy(weight_numpy).float().T
            else:
                print(f"ERROR: Variable '{tf_name}' not found in checkpoint!")
                return
        print("\nWeight extraction and mapping successful.")

    except Exception as e:
        print(f"ERROR while reading checkpoint: {e}")
        return

    torch.save(pytorch_state_dict, PYTORCH_WEIGHTS_PATH)
    print(f"Converted weights saved to: {PYTORCH_WEIGHTS_PATH}")

if __name__ == '__main__':
    if not tf.__version__.startswith('1.'):
        print("Error: TensorFlow 1.x required to run this conversion script directly.")
        exit()
    convert_tf1_to_pytorch()
