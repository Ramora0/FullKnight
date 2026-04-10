# Training Speed Improvements

Baseline: ~72s/epoch with 8 envs. Bottleneck is 31ms/step of JSON/WS overhead (31.7s/epoch) and 22ms/step of HK simulation (22.5s/epoch).

1. **Binary protocol** (~35%): Replace JSON serialization of hitbox arrays with MessagePack or raw float32 buffers. 31ms of per-step overhead drops to ~5ms, saving ~26s/epoch.

2. **Split-group pipelined stepping** (~17%): Split 8 envs into 2 groups of 4 and alternate, so GPU inference on one group overlaps with HK simulation of the other. Hides the 11ms forward pass entirely inside the 53ms HK wait.

3. **Reduce HK per-frame cost** (~10%): Add low resolution, disable VSync/audio/quality in the mod's Setup. Cuts the 22ms C# sim time per step to ~12-15ms.

4. **Overlap training with resets** (~8%): Run PPO training in a background thread while staggered resets do their WS round-trips. Makes the 6.24s training phase partially free.

5. **torch.compile** (~7%): One-line change fuses GPU kernels, reducing kernel launch overhead that dominates at batch_size=8. Bigger relative impact once other overhead shrinks.
