import matplotlib.pyplot as plt
import numpy as np

# Function to create a radar chart
def create_radar_chart(data, labels, title, ax, colors, model_names):
    num_vars = len(labels)

    # Compute angle for each axis
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    data = [d + d[:1] for d in data]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    # Draw ylabels with an even distribution from 0 to 100
    ax.set_rscale('linear')
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'])

    for idx, d in enumerate(data):
        ax.plot(angles, d, color=colors[idx], linewidth=2, linestyle='solid', label=model_names[idx])
        ax.fill(angles, d, color=colors[idx], alpha=0.2)

    ax.legend(loc='upper left', framealpha=1)

# Data for the models
labels_anomaly_gpt = ['Overall Accuracy', 'i_AUROC', 'p_AUROC']
data_anomaly_gpt = [
    [87.83, 94.90, 95.63],  # AnomalyGPT on MVTec Data
    [38.58, 67.25, 58.69]   # AnomalyGPT on Grape Leaf Data
]
model_names_anomaly_gpt = ['AnomalyGPT on MVTec', 'AnomalyGPT on Grape Leaves']

labels_other_models = ['Overall Accuracy', 'Healthy Recall', 'Esca Recall']
data_other_models = [
    [92, 90, 97],       # Traditional CNN
    [71, 90, 50],       # CLIP
    [84.16, 90, 82]     # KAN (assuming recall values)
]
model_names_other_models = ['Traditional CNN', 'CLIP', 'KAN']

# Create radar charts
fig, axs = plt.subplots(2, 1, figsize=(12, 16), subplot_kw=dict(polar=True))
colors_anomaly_gpt = ['#1f77b4', '#ff7f0e']
colors_other_models = ['#2ca02c', '#d62728', '#9467bd']

create_radar_chart(data_anomaly_gpt, labels_anomaly_gpt, 'AnomalyGPT Models Comparison', axs[0], colors_anomaly_gpt, model_names_anomaly_gpt)
#axs[0].set_title('AnomalyGPT Models Comparison')

create_radar_chart(data_other_models, labels_other_models, 'Other Models Comparison', axs[1], colors_other_models, model_names_other_models)
#axs[1].set_title('Other Models Comparison')

plt.show()
fig.savefig('/Users/michaelrodden/Desktop/radar_charts.png')
