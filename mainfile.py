from utils import *
from torch import backends
from torch import save
from torchsummary import summary
import os, gc, torch

def maybe_resume(configs, model, optimizer, device):
    """
    Resume training from a checkpoint path (if provided).
    """
    ckpt_path = getattr(configs, "checkpoint", None)
    if not ckpt_path:
        print("[resume] No checkpoint provided; starting fresh.")
        return 1  # start from scratch

    if not os.path.isfile(ckpt_path):
        print(f"[resume] Checkpoint '{ckpt_path}' not found; starting fresh.")
        return 1

    print(f"[resume] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    # Load model weights
    model.load_state_dict(ckpt["model_state_dict"], strict=False)

    # Load optimizer if available
    if "optimizer_state_dict" in ckpt and optimizer is not None:
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            print("[resume] Optimizer state loaded.")
        except Exception as e:
            print(f"[resume] Skipping optimizer state (reason: {e})")

    start_epoch = int(ckpt.get("epoch", 1))
    print(f"[resume] Resuming from epoch {start_epoch}")
    return start_epoch


def main(configs):
    model, optimizer, dataDims = initialize_training(configs)
    torch.cuda.empty_cache()

    print(f"Imported and Analyzed Training Dataset {configs.training_dataset}")

    if configs.max_epochs == 0:
        sample, time_ge = generation(configs, dataDims, model)
    else:
        epoch = maybe_resume(configs, model, optimizer, device="cuda:0")

        converged = 0
        tr_err_hist, te_err_hist = [], []
        tr, te, _ = get_dataloaders(configs)

        while (epoch <= (configs.max_epochs + 1)) and (converged == 0):
            err_tr, time_tr = model_epoch_new(
                configs, dataDims=dataDims, trainData=tr,
                model=model, optimizer=optimizer, update_gradients=True
            )
            err_te, time_te = model_epoch_new(
                configs, dataDims=dataDims, trainData=te,
                model=model, update_gradients=False
            )

            tr_err_hist.append(torch.mean(torch.stack(err_tr)))
            te_err_hist.append(torch.mean(torch.stack(err_te)))

            print(
                f"epoch={epoch}; "
                f"nll_tr={torch.mean(torch.stack(err_tr)):.5f}; "
                f"nll_te={torch.mean(torch.stack(err_te)):.5f}; "
                f"time_tr={time_tr:.1f}s; time_te={time_te:.1f}s"
            )

            converged = auto_convergence(configs, epoch, tr_err_hist, te_err_hist)

            if epoch % 2 == 0:
                train_loss = torch.mean(torch.stack(err_tr)).item()
                val_loss = torch.mean(torch.stack(err_te)).item()
                with open("training_process_info.txt", "a") as f:
                    f.write(f"Epoch {epoch}: Train Loss = {train_loss:.6f}, "
                            f"Val Loss = {val_loss:.6f}, "
                            f"Time = {time_tr:.2f}s\n")

            if epoch % configs.generation_period == 0:
                sample, time_ge = generation(configs, dataDims, model)

            if epoch % configs.save_checkpoints == 0:
                save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                }, f"./model-{epoch}.pt")

            epoch += 1

        save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, f"./model-{epoch}.pt")

        sample, time_ge = generation(configs, dataDims, model)

    print("finished!")
