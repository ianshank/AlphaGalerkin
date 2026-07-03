# C4 Mermaid Conventions — Skeletons and Checklist

## L1 — System Context Skeleton

```mermaid
C4Context
    title System Context — <System Name>

    Person(operator, "Operator", "Runs training jobs and reviews results")
    Person(consumer, "API Consumer", "Calls the inference endpoint")

    System(core_system, "<System Name>", "What the system does, one sentence")

    System_Ext(object_store, "Object Storage", "Holds datasets and checkpoints")
    System_Ext(experiment_tracker, "Experiment Tracker", "Records metrics and artifacts")

    Rel(operator, core_system, "launches and monitors runs via", "CLI")
    Rel(consumer, core_system, "requests predictions from", "HTTPS/JSON")
    Rel(core_system, object_store, "reads datasets from / writes checkpoints to", "SDK")
    Rel(core_system, experiment_tracker, "publishes metrics to", "HTTPS")
```

## L2 — Container Skeleton

```mermaid
C4Container
    title Containers — <System Name>

    Person(operator, "Operator", "Runs training jobs")

    System_Boundary(core_system, "<System Name>") {
        Container(cli, "Control CLI", "Python/argparse", "Entry point; validates config, dispatches jobs")
        Container(training_service, "Training Service", "Python/PyTorch", "Runs the training loop and self-play")
        Container(inference_service, "Inference Service", "Python/ONNX Runtime", "Serves the exported model")
        ContainerDb(checkpoint_store, "Checkpoint Store", "Filesystem/Object store", "Versioned model checkpoints")
    }

    System_Ext(experiment_tracker, "Experiment Tracker", "Metrics and artifacts")

    Rel(operator, cli, "invokes", "shell")
    Rel(cli, training_service, "launches with validated config", "subprocess")
    Rel(training_service, checkpoint_store, "writes checkpoints to", "atomic file ops")
    Rel(inference_service, checkpoint_store, "loads latest checkpoint from", "read-only")
    Rel(training_service, experiment_tracker, "streams metrics to", "HTTPS")
```

## L3 — Component Skeleton (one container)

```mermaid
C4Component
    title Components — Training Service

    Container_Boundary(training_service, "Training Service") {
        Component(config_loader, "Config Loader", "Pydantic", "Parses and validates run configuration")
        Component(trainer, "Trainer", "Python", "Owns the optimization loop; emits metrics")
        Component(replay_buffer, "Replay Buffer", "Python", "Stores and samples experience")
        Component(self_play, "Self-Play Worker", "Python/MCTS", "Generates games against the current model")
        Component(checkpoint_mgr, "Checkpoint Manager", "Python", "Atomic save/load with rotation")
    }

    ContainerDb(checkpoint_store, "Checkpoint Store", "Filesystem", "Versioned checkpoints")

    Rel(config_loader, trainer, "provides typed config to", "constructor injection")
    Rel(self_play, replay_buffer, "appends episodes to", "in-process")
    Rel(trainer, replay_buffer, "samples batches from", "in-process")
    Rel(trainer, checkpoint_mgr, "requests checkpoint via", "in-process")
    Rel(checkpoint_mgr, checkpoint_store, "persists to", "atomic write")
```

## Dynamic View — sequenceDiagram Template

```mermaid
sequenceDiagram
    title Key Flow — <flow name, e.g. "Resume training from checkpoint">

    actor Operator
    participant CLI as Control CLI
    participant Trainer as Training Service
    participant Store as Checkpoint Store

    Operator->>CLI: run --resume <checkpoint-id>
    CLI->>CLI: validate config (Pydantic)
    CLI->>Trainer: launch(config)
    Trainer->>Store: load(checkpoint-id)
    alt checkpoint found
        Store-->>Trainer: state dict + metadata
        Trainer->>Trainer: restore optimizer, step counter
    else missing / corrupt
        Store-->>Trainer: error
        Trainer-->>CLI: exit non-zero with reason
    end
    loop each training step
        Trainer->>Trainer: step()
    end
    Trainer->>Store: save(new checkpoint)
    Trainer-->>CLI: final metrics
    CLI-->>Operator: summary + exit 0
```

Rules for dynamic views:
- One flow per diagram; name the flow in the title.
- Show error/fallback branches with `alt`/`else` when the flow has them — a happy-path-only diagram of a retrying flow is wrong.
- Participants map 1:1 to containers (or components, if the flow is intra-container). Do not invent participants that have no C4 element.

## Naming Conventions

| Item | Convention | Example |
|---|---|---|
| Alias | snake_case, stable across levels | `training_service` |
| Label | Title Case, human-readable | `Training Service` |
| Technology arg | concrete tech, slash-separated | `Python/PyTorch` |
| Rel label | active verb phrase | `writes checkpoints to` |
| Rel technology | protocol or mechanism | `HTTPS/JSON`, `in-process` |
| Diagram title | `<Level> — <Scope>` | `Components — Training Service` |

## Review Checklist

- [ ] Correct block type per level (`C4Context`/`C4Container`/`C4Component`), one diagram per block.
- [ ] ≤ ~10 elements; split by boundary if over.
- [ ] Exactly one boundary nesting level per diagram.
- [ ] Every element has a non-empty description stating its single responsibility.
- [ ] Every `Rel` has a verb-phrase label and a technology argument.
- [ ] Rel direction = call direction (use `BiRel` only with a dual-action label).
- [ ] Aliases snake_case and consistent with other levels.
- [ ] Externals use `System_Ext` and sit outside the boundary.
- [ ] Each headline flow has a `sequenceDiagram` with error branches where they exist.
- [ ] Prose summary above each diagram; updated in the same PR as the structural change.
