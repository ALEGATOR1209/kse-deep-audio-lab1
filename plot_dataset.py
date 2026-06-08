import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def plot_gender(df, ax=None, suffix=''):
    if ax is None:
        fig, ax = plt.subplots()

    by_sex = df.groupby('sex').size().reset_index(name='counts')
    colors = plt.colormaps['Pastel1'].colors[:len(by_sex)]
    ax.bar(by_sex['sex'], by_sex['counts'], color=colors)
    ax.set_title(f'Gender distribution{suffix}')
    ax.bar_label(ax.containers[0], fmt='%d')


def plot_emotion(df, ax=None, suffix=''):
    if ax is None:
        fig, ax = plt.subplots()

    by_emotion = df.groupby('emotion').size().reset_index(name='counts')
    colors = plt.colormaps['Pastel1'].colors[:len(by_emotion)]
    ax.bar(by_emotion['emotion'], by_emotion['counts'], color=colors)
    ax.set_title(f'Emotional distribution{suffix}')
    ax.bar_label(ax.containers[0], fmt='%d')


def plot_duration(df, ax=None, suffix=''):
    if ax is None:
        fig, ax = plt.subplots()

    by_duration = df.groupby('duration_seconds').size().reset_index(name='counts')
    ax.bar(by_duration['duration_seconds'], by_duration['counts'])
    ax.set_title(f'Samples by duration{suffix}')
    ax.set_xlabel('Duration, s')
    ax.set_ylabel('Samples, n')


def plot_words(df, ax=None, suffix=''):
    if ax is None:
        fig, ax = plt.subplots()

    by_words = df.groupby('nwords').size().reset_index(name='counts')
    ax.bar(by_words['nwords'], by_words['counts'])
    ax.set_title(f'Samples by words{suffix}')
    ax.set_xlabel('Words in a sample, n')
    ax.set_ylabel('Samples, n')


def plot_sex_emotion(df, ax=None, suffix=''):
    if ax is None:
        fig, ax = plt.subplots()

    by_sex_emotion = df.groupby(['sex', 'emotion']).size().reset_index(name='counts')
    by_sex_emotion['index'] = by_sex_emotion['sex'].combine(by_sex_emotion['emotion'], lambda x, y: f"{x}_{y}")

    colors = plt.colormaps['Pastel1'].colors[:len(set(by_sex_emotion))]
    ax.bar(by_sex_emotion['index'], by_sex_emotion['counts'], color=colors)
    ax.set_title(f'Emotional and sex distribution{suffix}')
    ax.bar_label(ax.containers[0], fmt='%d')
    ax.tick_params(axis='x', rotation=45)


def plot_duration_by_sex(df, ax=None, suffix=''):
    if ax is None:
        fig, ax = plt.subplots()

    by_sex_duration = df.groupby('sex')['duration_seconds'].apply(list).to_dict()
    sex_colors = {s: plt.colormaps['Pastel1'].colors[i] for i, s in enumerate(df['sex'].unique())}

    for sex, durations in by_sex_duration.items():
        ax.hist(
            durations,
            bins=30,
            label=sex,
            facecolor=(*sex_colors[sex][:3], 0.5),
            edgecolor=sex_colors[sex],
            linewidth=0.8,
        )

    ax.legend(title='Sex', framealpha=0.7)
    ax.set_title(f'Sample duration distribution by sex{suffix}')
    ax.set_xlabel('Duration (s)')
    ax.set_ylabel('Count')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3, linestyle='--')


def plot_words_vs_duration(df, ax=None, suffix=''):
    if ax is None:
        fig, ax = plt.subplots()

    words_per_second = df.copy()
    words_per_second['wps'] = words_per_second['nwords'] / words_per_second['duration_seconds']
    ax.scatter(words_per_second['duration_seconds'], words_per_second['nwords'])
    ax.set_title(f'Distribution of number of words per duration{suffix}')
    ax.set_xlabel('Duration (s)')
    ax.set_ylabel('Words, n')
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))


def plot_dataset(df, suffix=''):
    fig, ax = plt.subplots(4, 2, figsize=(10, 20))

    plot_gender(df, ax[0, 0], suffix)
    plot_emotion(df, ax[0, 1], suffix)
    plot_duration(df, ax[1, 0], suffix)
    plot_words(df, ax[1, 1], suffix)
    plot_sex_emotion(df, ax[2, 0], suffix)
    plot_duration_by_sex(df, ax[2, 1], suffix)
    plot_words_vs_duration(df, ax[3, 0], suffix)
    ax[3, 1].set_visible(False)

    fig.tight_layout()
    return fig
