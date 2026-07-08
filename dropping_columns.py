def train_and_evaluate(events, num_epochs=200, k_folds=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    all_losses = []
    all_accuracies = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(events)):
        print(f"\nFold {fold + 1}/{k_folds}")
        train_events = [events[i] for i in train_idx]
        test_events = [events[i] for i in test_idx]

        model = BiGCN(input_dim=5000, hidden_dim=128, lstm_hidden_dim=64, output_dim=4, num_layers=1, dropedge_rate=0.3, dropout_rate=0.5).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        model.train()
        losses = []
        for epoch in range(num_epochs):
            total_loss = 0
            for event in train_events:
                features = event['features'].to(device)
                adj_matrix = event['adj_matrix'].to(device)
                label = event['label'].to(device)
                root_idx = event['root_idx']

                optimizer.zero_grad()
                output = model(features, adj_matrix, root_idx)
                loss = criterion(output.unsqueeze(0), label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_events)
            losses.append(avg_loss)
            all_losses.append(avg_loss)

            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {avg_loss:.4f}")

            if epoch > 10 and avg_loss < 0.01:
                print("Early stopping triggered.")
                break

        model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for event in test_events:
                features = event['features'].to(device)
                adj_matrix = event['adj_matrix'].to(device)
                label = event['label'].to(device)
                root_idx = event['root_idx']

                output = model(features, adj_matrix, root_idx)
                pred = output.argmax().item()
                preds.append(pred)
                true_labels.append(label.item())

        acc = accuracy_score(true_labels, preds)
        f1_scores = f1_score(true_labels, preds, average=None)
        fold_results.append({'acc': acc, 'f1': f1_scores})
        all_accuracies.append(acc)
        print(f"Fold {fold + 1} - Accuracy: {acc:.4f}, F1 Scores: {f1_scores}")

    avg_acc = np.mean([r['acc'] for r in fold_results])
    avg_f1 = np.mean([r['f1'] for r in fold_results], axis=0)
    print(f"\nAverage Accuracy: {avg_acc:.4f}")
    print(f"Average F1 Scores (N, F, T, U): {avg_f1}")

    # Visualization
    plt.figure(figsize=(12, 5))

    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(all_losses, label='Loss per Epoch')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Accuracy curve
    plt.subplot(1, 2, 2)
    plt.plot(all_accuracies, label='Accuracy per Fold')
    plt.title('Accuracy Curve')
    plt.xlabel('Fold')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_metrics.png')

