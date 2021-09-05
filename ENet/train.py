from tqdm import tqdm

def train(dataloader, model, loss_fn, optimizer, device="cpu"):
    train_losses = []
    loader = tqdm(dataloader)
    loader.set_description("Training...")
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        pred = model(data)
        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loader.set_postfix(train_batch_loss=loss.item())
        train_losses.append(loss.item())

    return sum(train_losses) / len(train_losses)