# plots.py
from reporting import load_experiment_summary, plot_precision_comparison, plot_unlearning_time, plot_num_groups_vs_precision


def main():
    df = load_experiment_summary('experiment_results.csv')
    plot_precision_comparison(df)
    plot_unlearning_time(df)
    plot_num_groups_vs_precision(df)
    print('Saved reusable plots to plots/*.png')


if __name__ == '__main__':
    main()
