import argparse
import pathlib
import subprocess
import sys
from collections import namedtuple

CMISettings = namedtuple('CMISettings',
                         ['name', 'expectation_samples', 'mixture_samples',
                          'reuse_action_samples', 'kl_type', 'add_entropy'])
EntropySettings = namedtuple('EntropySettings',
                             ['name', 'expectation_samples'])
ContactSettings = namedtuple('ContactSettings', ['name'])
AttentionSettings = namedtuple('AttentionSettings', ['name', 'factorizer'])

SETTINGS = {
    'var_prod_approx': ([CMISettings(f'kl_approx_{n}', n, n, True,
                                     'var_prod_approx', False)
                         for n in [8, 16, 32, 64]] +
                        [CMISettings(f'kl_approx_{n}_{n}', n, n, False,
                                     'var_prod_approx', False)
                         for n in [8, 16, 32, 64]]),
    'entropy': [EntropySettings(f'entropy_{n}', n)
                for n in [8, 16, 32, 64]],
    'contact': [ContactSettings('contacts')],
    'attention_onedslide': [AttentionSettings('attention',
                                              '@factorization.one_d_slide')],
    'attention_fetchpnp': [AttentionSettings(
                            'attention',
                            '@factorization.fetch_with_object')]
}

DEFAULT_SETTINGS = ('var_prod_approx', 'entropy')

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--verbose', action='store_true', help='More info')
parser.add_argument('--seed', type=int, default=101, help='Seed for RNG')
parser.add_argument('-o', '--output-path', type=pathlib.Path,
                    help='Path to output directory')
parser.add_argument('--model-path', type=pathlib.Path,
                    help='Path to model directory')
parser.add_argument('--settings-path', type=pathlib.Path,
                    help='Path to settings file')
parser.add_argument('--memory-path', type=pathlib.Path,
                    help='Path to memory buffer')
parser.add_argument('--extra-args', nargs='+',
                    help='Extra arguments to eval script')
parser.add_argument('--variants', nargs='*',
                    default=DEFAULT_SETTINGS,
                    choices=list(SETTINGS),
                    help='Variants to run')


def format_config_args(s):
    if isinstance(s, CMISettings):
        args = [f'main.scorer_cls=@CMIScorer',
                f'CMIScorer.n_expectation_samples={s.expectation_samples}',
                f'CMIScorer.n_mixture_samples={s.mixture_samples}',
                f'CMIScorer.reuse_action_samples={s.reuse_action_samples}',
                f'CMIScorer.kl_type="{s.kl_type}"',
                f'CMIScorer.add_entropy={s.add_entropy}']
    elif isinstance(s, EntropySettings):
        args = [f'main.scorer_cls=@EntropyScorer',
                f'EntropyScorer.n_expectation_samples={s.expectation_samples}']
    elif isinstance(s, ContactSettings):
        args = [f'main.scorer_cls=@ContactScorer']
    elif isinstance(s, AttentionSettings):
        args = [f'main.scorer_cls=@MaskScorer',
                f'MaskScorer.state_factorizer={s.factorizer}']

    return args


def run_eval_influence(output_path, data_path, model_path,
                       settings_path, config_args, seed=0, verbose=False):
    proj_dir = pathlib.Path(__file__).parent.parent.parent

    py_cmd = ['python', '-m',
              'cid.influence_estimation.eval_influence',
              f'--seed={seed}',
              f'--output-path={output_path}',
              str(data_path),
              str(model_path),
              str(settings_path)] + config_args

    if verbose:
        print(' '.join(py_cmd))

    try:
        res = subprocess.run(py_cmd, check=True, cwd=str(proj_dir),
                             capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        print(e.stderr)
        raise e

    return res.stdout


if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    args.output_path.mkdir(parents=True, exist_ok=True)

    selected_settings = [s for name, settings in SETTINGS.items()
                         for s in settings if name in args.variants]
    output = ''
    for settings in selected_settings:
        config_args = format_config_args(settings)
        if args.extra_args is not None:
            config_args += args.extra_args
        log = run_eval_influence(args.output_path / f'{settings.name}.npy',
                                 args.memory_path,
                                 args.model_path,
                                 args.settings_path,
                                 config_args,
                                 args.seed,
                                 args.verbose)
        output += log
        print(log)

    with open(args.output_path / 'log_scoring.txt', 'w') as f:
        f.write(output)
