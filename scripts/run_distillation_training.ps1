param(
    [int]$TargetExamples = 300,
    [string]$TeacherModel = "Qwen/Qwen2.5-7B-Instruct:fastest",
    [string]$BaseModel = "Qwen/Qwen2.5-0.5B-Instruct",
    [string]$OutputDir = "models\socimobai-distilled-qwen",
    [double]$Epochs = 4,
    [double]$LearningRate = 0.00005
)

if (-not $env:HF_TOKEN) {
    throw "Defina HF_TOKEN antes de rodar. Exemplo: `$env:HF_TOKEN='seu_token'"
}

python scripts\generate_teacher_dataset.py `
    --target-examples $TargetExamples `
    --model $TeacherModel `
    --output data\teacher_socimob_qwen7b.jsonl

python scripts\train_finetune.py `
    --dataset data\treinamento_imobiliaria.jsonl data\treinamento_socimob_regras.jsonl data\teacher_socimob_qwen7b.jsonl `
    --base-model $BaseModel `
    --epochs $Epochs `
    --learning-rate $LearningRate `
    --output-dir $OutputDir

python scripts\test_finetuned_model.py `
    --model-dir $OutputDir `
    --message "HISTORICO:`nSem historico anterior.`n`nDADOS DO LEAD:`n- objetivo_compra: aluguel`n- preferencia_bairro: Santa Luzia`n- budget_max: 1500`n`nIMOVEIS REAIS DISPONIVEIS:`nNenhum imovel real enviado.`n`nMENSAGEM ATUAL DO CLIENTE:`nQuero alugar apartamento ate 1500 em Santa Luzia`n`nResponda como Teresa:"
