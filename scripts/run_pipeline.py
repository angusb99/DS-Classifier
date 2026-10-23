"""Run the dscm_run.ipynb workflow end to end against the local data/ folder."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from functions import training_functions as tf  # noqa: E402

CLASS_NAMES = ['OTHER', 'x11', 'x21', 'x22', 'x31', 'x32', 'x33', 'x34', 'x35', 'x91', 'x92', 'x93', 'x94']

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--period", default="a25")
parser.add_argument("--state", default="SA")
parser.add_argument("--pool", default="all")
parser.add_argument("--method", default="building_target")
parser.add_argument("--extended-features", action=argparse.BooleanOptionalAction, default=True)
parser.add_argument("--adhoc-output", help="folder (relative to data/output, or absolute) to write all outputs to instead of the quarter folders")
parser.add_argument("--collapse-other-classes", action=argparse.BooleanOptionalAction, default=True,
                    help="merge x35/x91-x94 into OTHER before training (--no-collapse-other-classes keeps them)")
args = parser.parse_args()

tf.set_output_location(args.adhoc_output)
tf.start_log(args.state)

start_time = time.perf_counter()
tf.generate_training_data_workflow(args.period, args.state, args.pool, args.method, args.extended_features)
print(f"Full time for training data generation {(time.perf_counter() - start_time):.4f} seconds.")

start_time = time.perf_counter()
model, X_val, y_val, le, class_set = tf.train_test_model(args.period, args.state, args.method, args.collapse_other_classes)
print(f"Full time for training model {(time.perf_counter() - start_time):.4f} seconds.")
class_names = [name for name in CLASS_NAMES if name in class_set]

start_time = time.perf_counter()
model_eval_output = tf.complete_model_evaluation(
    args.period, args.state, args.method, model,
    X_val.drop(columns='ARID'), y_val[['DWELLING_STRUCT']], class_names=class_names
)
print(f"Full time for obtaining model evaluation {(time.perf_counter() - start_time):.4f} seconds.")
print(f"Validation accuracy: {model_eval_output['metrics']['accuracy']:.4f}, "
      f"macro F1: {model_eval_output['metrics']['f1_macro']:.4f}")

tf.model_test_predictions(args.period, args.state, args.method, model_eval_output, X_val, y_val, class_names=class_names, le=le)
tf.predict_candidates(args.period, args.state, args.method, model, class_set=class_names, pool=args.pool)
