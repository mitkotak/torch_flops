import lovelyplots
import matplotlib.pyplot as plt
plt.style.use(['ipynb'])
import numpy as np


data = {
    'resnet50': {
        2: {
            'analytical_time': 0.973,
            'empirical_time': 5.03760290145874
        },
        4: {
            'analytical_time': 1.946,
            'empirical_time': 5.036628723144531
        },
        8: {
            'analytical_time': 3.891,
            'empirical_time': 6.354036331176758
        },
        16: {
            'analytical_time': 7.782,
            'empirical_time': 11.954041481018066
        },
        32: {
            'analytical_time': 15.564,
            'empirical_time': 22.195842742919922
        },
        64: {
            'analytical_time': 31.128,
            'empirical_time': 41.840938568115234
        }
    },
    'vit_base16': {
        2: {
            'analytical_time': 4.158,
            'empirical_time': 7.987310409545898
        },
        4: {
            'analytical_time': 8.305,
            'empirical_time': 13.127836227416992
        },
        8: {
            'analytical_time': 16.599,
            'empirical_time': 24.562789916992188
        },
        16: {
            'analytical_time': 33.186,
            'empirical_time': 47.97847366333008
        },
        32: {
            'analytical_time': 66.361,
            'empirical_time': 93.92566680908203
        },
        64: {
            'analytical_time': 132.711,
            'empirical_time': 185.07034301757812
        }
    }
}


# Assuming the performance_data dictionary is already defined

def extract_data(data, key):
    channels = sorted(data['resnet50'].keys())
    resnet50_values = [data['resnet50'][ch][key] for ch in channels]
    vit_base16_values = [data['vit_base16'][ch][key] for ch in channels]
    return channels, resnet50_values, vit_base16_values

def create_plot(x, y1, y2, title, ylabel, legend_labels):
    plt.plot(x, y1, label=legend_labels[0])
    plt.plot(x, y2, label=legend_labels[1])
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.xlabel('channels')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=5)
    plt.grid('--')
    plt.tight_layout()

# Extract data
channels, resnet50_analytical, vit_base16_analytical = extract_data(data, 'analytical_time')
_, resnet50_empirical, vit_base16_empirical = extract_data(data, 'empirical_time')

# Create analytical time plot
create_plot(
    channels,
    resnet50_analytical,
    vit_base16_analytical,
    'Analytical Runtime vs Channels',
    'Time (ms)',
    ['ResNet50 AI=36.48', 'ViT-Base16 AI=197.69']
)
plt.ylim(10**-0.5, 10**2.5)
plt.savefig('timm_analytical_inference.png', transparent=False)
plt.close()

# Create empirical time plot
create_plot(
    channels,
    resnet50_empirical,
    vit_base16_empirical,
    'Empirical Runtime vs Channels',
    'Time (ms)',
    ['ResNet50', 'ViT-Base16']
)
plt.ylim(10**-0.5, 10**2.5)
plt.savefig('timm_empirical_inference.png', transparent=False)
plt.close()
