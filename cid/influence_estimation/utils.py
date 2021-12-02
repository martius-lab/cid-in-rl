import pathlib


def convert_train_settings(train_settings_path,
                           output_path=None,
                           preamble=None,
                           prefixes=None):
    """Make Gin eval settings file from training settings file"""
    settings_path = pathlib.Path(train_settings_path)
    if output_path is None:
        output_path = settings_path.parent / f'eval_{settings_path.name}'
    else:
        output_path = pathlib.Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

    if prefixes is None:
        prefixes = [
            'import',
            'main.episode_len',
            'ForwardGoalDataset',
            'FactorizedForwardDataset',
            'GaussianLikelihoodHead',
            'MLP',
            'Transformer'
        ]

    if preamble is None:
        out_lines = [
            'import cid.influence_estimation'
        ]
    else:
        out_lines = preamble

    with open(settings_path, 'r') as f:
        saved_line = None
        for line in f.readlines():
            line = line.rstrip()
            if saved_line is not None:
                out_lines.append(saved_line + line)
                saved_line = None
                continue

            if line.startswith('#'):
                out_lines.append('')
            elif line.endswith('\\'):
                saved_line = line[:-1]
            elif (line.startswith('main.dataset_classes')
                    or line.startswith('main.model_classes')):
                out_lines.append(line.replace('main',
                                              'make_pretrained_scorer'))
            else:
                for prefix in prefixes:
                    if line.startswith(prefix) and line not in out_lines:
                        out_lines.append(line)
                        break

    with open(output_path, 'w') as f:
        f.write('\n'.join(out_lines))

    return output_path
