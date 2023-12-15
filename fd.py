import ray
from ray import tune
from ray.train import report  # Import the new report function

def train_tune(config):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MiniNet().to(device)  # Replace with your actual model implementation

    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=config["momentum"], weight_decay=config["weight_decay"])

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)

    triplet_criterion = torch.nn.TripletMarginLoss()

    for epoch in range(3):  # Adjust the number of epochs as needed
        mini_net.train()
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            optimizer.zero_grad()
            anchor_outputs = model(anchor)
            positive_outputs = model(positive)
            negative_outputs = model(negative)
            loss = triplet_criterion(anchor_outputs, positive_outputs, negative_outputs)
            loss.backward()
            optimizer.step()

        scheduler.step()

        mini_net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_batch_idx, (val_anchor, val_positive, val_negative) in enumerate(val_loader):
                val_anchor_outputs = mini_net(val_anchor)
                val_positive_outputs = mini_net(val_positive)
                val_negative_outputs = mini_net(val_negative)
                val_loss += triplet_criterion(val_anchor_outputs, val_positive_outputs, val_negative_outputs).item()

        average_val_loss = val_loss / len(val_loader)

        report(mean_loss=average_val_loss)  # Use the new report function with 'mean_loss' argument

if _name_ == "_main_":
    ray.init()

    # Manually specify fewer combinations
    reduced_config = {
        "lr": 0.0001,
        "momentum": 0.9,
        "weight_decay": 0.0001
    }

    analysis = tune.run(
        train_tune,
        config=reduced_config,
        resources_per_trial={"cpu": 0, "gpu": 1},
        num_samples=4,  # Adjust the number of samples as needed
        progress_reporter=tune.CLIReporter()
    )

    best_config = analysis.get_best_config(metric="mean_loss", mode="min")
    print("Best config:", best_config)

    ray.shutdown()