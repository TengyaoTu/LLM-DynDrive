##--------Our work-----------------
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -u transformer_inference_steer_dp.py \
    --model_name_or_path '/data1/yourfolder/models/QwQ-32B' \
    --dataset_dir "./Data/" \
    --output_path "./outputs_steer_dynamic" \
    --dataset "Math_Math500" \
    --max_generated_tokens 8000 \
    --num_gpus 8 \
    --steer_vector_path ./outputs/QwQ-32B/steer_vector_layer58_conf_mixed.pt \
    --steer_layer 58 \
    --steer_coef -1
##----------Extract Hidden State----
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python transformer_inference_dp.py \
  --model_name_or_path '/data1/tutengyao/models/QwQ-32B' \
  --dataset_dir "./Data" \
  --dataset "Math_Olympiad" \
  --output_path "./outputs" \
  --max_generated_tokens 10000 \
  --num_gpus 8 \
  --trust_remote_code
##------------Calculate vector and get the parameter
python hidden_analysis_mixed_auto.py \
  --layer_id 58 \
  --jsonl_path ./outputs/QwQ-32B/Math_Math500/origin_temp0.7_maxlen16000.merged.jsonl \
  --hidden_dir ./outputs/QwQ-32B/Math_Math500/ \
  --save_path  ./outputs/QwQ-32B/steer_vector_layer58_conf_mixed.pt \
  --threshold 0.70 \
  --max_files 500 \
  --expected_offset 1
##-----------merge shards--------------
python merge_shards.py \
  --dir ./outputs/QwQ-32B/Math_Math500\
  --base 'origin_temp0.7_maxlen8000'
##-----------evaluate results----------
python check.py \
    --model_name_or_path '/data1/tutengyao/models/QwQ-32B' \
    --data_name "Math_Math500" \
    --generation_path "./outputs/QwQ-32B/Math_Math500/origin_temp0.7_maxlen8000.merged.jsonl"


  
