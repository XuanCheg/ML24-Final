from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftType,
)
import torch
from transformers import (
    LLaMaForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import transformers
from utils.train_utils import compute_metrics


def setup_model(args):
    peft_config = LoraConfig(task_type=TaskType.TASK_CLS, 
                             peft_type=PeftType.Lora,)

    num_labels = len(args.label_id[args.dset_name])
    print('*********************')
    print(num_labels)
    print('*********************')
    model = LLaMaForSequenceClassification.from_pretrained(args.model_name_or_path, 
                                                           return_dict=True, num_labels=num_labels)
    model.config.pad_token_id = model.config.eos_token_id
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    model = model.half()
    return model

def setup_trainer(args, model, train_dataset, eval_dataset):
    training_args = TrainingArguments(
        output_dir=args.results_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.eval_bsz,
        dataloader_pin_memory=args.pin_memory,
        dataloader_num_workers=args.num_workers,
        num_train_epochs=args.n_epoch,
        weight_decay=args.wd,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=10,
        load_best_model_at_end=True,
        logging_dir=args.results_dir,
        logging_steps=1,
        report_to=["tensorboard"],
    )
    

    trainer = Trainer(
        model=model,
        args=training_args,
        # optimizers=(optimizer,linear_scheduler),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,   
        data_collator=train_dataset.collate_fn,
        compute_metrics=compute_metrics(args),
    )
    return trainer