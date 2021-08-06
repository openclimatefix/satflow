import numpy as np
import matplotlib.pyplot as plt

bands = [
    "HRV",
    "IR016",
    "IR039",
    "IR087",
    "IR097",
    "IR108",
    "IR120",
    "IR134",
    "VIS006",
    "VIS008",
    "WV062",
    "WV073",
]

baseline = np.load("baseline_current_image_mse_loss_channels_avg_base_only.npy")
flow = np.load("optical_flow_mse_loss_channels_avg_pairs.npy")
ssim = np.load("optical_flow_ssim_loss_channels_two_frames.npy")
ssim_base = np.load("baseline_current_image_ssim_loss_channels_baseline.npy")


# Now slice it up by channel and by timestep
per_channel_base = np.mean(baseline, axis=1)
per_channel_total = np.mean(flow, axis=1)
plt.plot(per_channel_base, label="Current Image")
plt.plot(per_channel_total, label="Optical Flow")
plt.legend(loc="best")
plt.xlabel("Channel #")
plt.ylabel("MSE")
plt.title("Mean MSE per channel")
plt.savefig("mse_per_channel.png", dpi=300)
plt.show()
per_channel_base = 1.0 - np.mean(ssim_base, axis=1)
per_channel_total = 1.0 - np.mean(ssim, axis=1)
plt.plot(per_channel_base, label="Current Image")
plt.plot(per_channel_total, label="Optical Flow")
plt.legend(loc="best")
plt.xlabel("Channel #")
plt.ylabel("SSIM")
plt.title("Mean SSIM per channel")
plt.savefig("ssim_per_channel.png", dpi=300)
plt.show()

per_channel_base = np.mean(baseline, axis=0)
per_channel_total = np.mean(flow, axis=0)
plt.plot(per_channel_base, label="Current Image")
plt.plot(per_channel_total, label="Optical Flow")
plt.legend(loc="best")
plt.xlabel("Timestep")
plt.ylabel("MSE")
plt.title("Mean MSE per timestep")
plt.savefig("mse_per_timestep.png", dpi=300)
plt.show()
per_channel_base = 1.0 - np.mean(ssim_base, axis=0)
per_channel_total = 1.0 - np.mean(ssim, axis=0)
plt.plot(per_channel_base, label="Current Image")
plt.plot(per_channel_total, label="Optical Flow")
plt.legend(loc="best")
plt.xlabel("Timestep")
plt.ylabel("SSIM")
plt.title("Mean SSIM per timestep")
plt.savefig("ssim_per_timestep.png", dpi=300)
plt.show()

for i in range(12):
    per_channel_base = baseline[i]
    per_channel_total = flow[i]
    plt.plot(per_channel_base, label="Current Image")
    plt.plot(per_channel_total, label="Optical Flow")
    plt.legend(loc="best")
    plt.xlabel("Timestep")
    plt.ylabel("MSE")
    plt.title(f"Mean MSE per timestep ({bands[i]})")
    plt.savefig(f"mse_{bands[i]}.png", dpi=300)
    plt.show()

for i in range(12):
    per_channel_base = 1.0 - ssim_base[i]
    per_channel_total = 1.0 - ssim[i]
    plt.plot(per_channel_base, label="Current Image")
    plt.plot(per_channel_total, label="Optical Flow")
    plt.legend(loc="best")
    plt.xlabel("Timestep")
    plt.ylabel("SSIM")
    plt.title(f"Mean SSIM per timestep ({bands[i]})")
    plt.savefig(f"ssim_{bands[i]}.png", dpi=300)
    plt.show()
