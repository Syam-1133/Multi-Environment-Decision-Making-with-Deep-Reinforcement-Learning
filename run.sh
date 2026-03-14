#!/bin/bash

# Helper script to run the Multi-Environment Decision Making project
# Usage: ./run.sh [train|test|tensorboard]

PYTHON="/Users/syamgudipudi/Desktop/Multi-Environment-Decision-Making--main/.venv/bin/python"
PROJECT_DIR="/Users/syamgudipudi/Desktop/Multi-Environment-Decision-Making--main/Multi-Env-Decision-Making-main"

cd "$PROJECT_DIR"

case "$1" in
    train)
        echo "🚗 Starting training on all 3 environments..."
        $PYTHON run.py --config config.yaml
        ;;
    train-highway)
        echo "🛣️  Training on Highway environment only..."
        $PYTHON run.py --config configurations/ddqn/ddqn_highway.yaml
        ;;
    train-merge)
        echo "🔀 Training on Merge environment only..."
        $PYTHON run.py --config configurations/ddqn/ddqn_merge.yaml
        ;;
    train-roundabout)
        echo "🔄 Training on Roundabout environment only..."
        $PYTHON run.py --config configurations/ddqn/ddqn_roundabout.yaml
        ;;
    test)
        if [ -z "$2" ]; then
            echo "❌ Please provide path to model: ./run.sh test <model_path>"
            exit 1
        fi
        echo "🧪 Testing model: $2"
        $PYTHON run.py --config config.yaml -m test -p "$2"
        ;;
    test-video)
        if [ -z "$2" ]; then
            echo "❌ Please provide path to model: ./run.sh test-video <model_path>"
            exit 1
        fi
        echo "🎥 Testing model with video: $2"
        $PYTHON run.py --config config.yaml -m test -p "$2" --render_video
        ;;
    tensorboard)
        echo "📊 Starting TensorBoard..."
        echo "Open http://localhost:6006 in your browser"
        $PYTHON -m tensorboard.main --logdir experiments/
        ;;
    help|*)
        echo "Multi-Environment Decision Making - Helper Script"
        echo ""
        echo "Usage: ./run.sh [command]"
        echo ""
        echo "Commands:"
        echo "  train              Train on all 3 environments (Highway + Merge + Roundabout)"
        echo "  train-highway      Train on Highway environment only"
        echo "  train-merge        Train on Merge environment only"
        echo "  train-roundabout   Train on Roundabout environment only"
        echo "  test <model>       Evaluate a trained model"
        echo "  test-video <model> Evaluate a trained model with video rendering"
        echo "  tensorboard        Start TensorBoard to monitor training"
        echo "  help               Show this help message"
        echo ""
        echo "Examples:"
        echo "  ./run.sh train"
        echo "  ./run.sh test experiments/ddqn/ddqn.pt"
        echo "  ./run.sh tensorboard"
        ;;
esac
