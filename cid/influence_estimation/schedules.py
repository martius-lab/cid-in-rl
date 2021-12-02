import gin


@gin.configurable(blacklist=['step'])
def batch_train_schedule(step, train_model_every,
                         buffer_warmup=200,
                         episode_len=50,
                         stop_after=10200,
                         p_latest_episodes=0.25,
                         multiplier_initial=8,
                         multiplier_start=2,
                         multiplier_interm=2,
                         multiplier_end=1,
                         rescore_initial=None,
                         rescore=None):
    if step >= stop_after:
        return False, None
    elif (step + 1 - buffer_warmup) % train_model_every != 0:
        return False, None

    samples = train_model_every * episode_len
    if step < 200:
        args = dict(batch_mode=True,
                    epochs_to_train=int(samples * multiplier_initial),
                    p_latest_episodes=None,
                    n_latest_episodes=None)
        rescore = rescore_initial
    elif step < 1200:
        args = dict(batch_mode=True,
                    epochs_to_train=int(samples * multiplier_start),
                    p_latest_episodes=None,
                    n_latest_episodes=None)
    elif step < 5200:
        args = dict(batch_mode=True,
                    epochs_to_train=int(samples * multiplier_interm),
                    p_latest_episodes=p_latest_episodes,
                    n_latest_episodes=train_model_every)
    else:
        args = dict(batch_mode=True,
                    epochs_to_train=int(samples * multiplier_end),
                    p_latest_episodes=p_latest_episodes,
                    n_latest_episodes=train_model_every)

    if rescore is not None:
        args['rescore_transitions'] = rescore

    return True, args
