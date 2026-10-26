def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument("--checkpoint", default="dqn.pth")
    parser.add_argument("--log_file", default="training.csv")

    parser.add_argument("--total_steps", type=int, default=3_000_000)
    parser.add_argument("--buffer_size", type=int, default=50_000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_starts", type=int, default=10_000)

    parser.add_argument("--train_frequency", type=int, default=4)
    parser.add_argument("--target_update", type=int, default=10_000)
    parser.add_argument("--save_frequency", type=int, default=50_000)

    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--gamma", type=float, default=0.99)

    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.1)
    parser.add_argument("--epsilon_decay", type=int, default=1_000_000)
    parser.add_argument("--test_epsilon", type=float, default=0.01)

    return parser


