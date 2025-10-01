#!/usr/bin/env bash
# grid_search.sh
PYTHON_EXEC=python

TRAIN_SCRIPT=transformer_unet_clean.py

declare -a DATA_DIRS=("data")

EPOCHS=50

declare -a LRS=(1e-4)
declare -a BATCH_SIZES=(4)
declare -a MAX_SEQS=(10)
declare -a HIDDENS=(1024)
declare -a W_CES=(1.0)            
declare -a W_DICES=(1)
declare -a W_JACCARDS=(1)
declare -a W_SHAPE_MONOS=(0)
declare -a W_DEPTH_MONOS=(1)
declare -a W_EXPANDS=(1)


for DATA_DIR in "${DATA_DIRS[@]}"; do

  
  OUTPUT_ROOT="./${DATA_DIR}_hyperparam_search_results"
  mkdir -p "${OUTPUT_ROOT}"

 
  for LR in "${LRS[@]}"; do
    for BS in "${BATCH_SIZES[@]}"; do
      for MS in "${MAX_SEQS[@]}"; do
        for HD in "${HIDDENS[@]}"; do
          for WCE in "${W_CES[@]}"; do
            for WD in "${W_DICES[@]}"; do
              for WJ in "${W_JACCARDS[@]}"; do
                for WSM in "${W_SHAPE_MONOS[@]}"; do
                  for WDM in "${W_DEPTH_MONOS[@]}"; do
                    for WE in "${W_EXPANDS[@]}"; do

                      
                      SUBDIR="lr_${LR}_bs_${BS}_seq_${MS}_hd_${HD}_wCE_${WCE}_wD_${WD}_wJ_${WJ}_wSM_${WSM}_wDM_${WDM}_wE_${WE}"
                      EXP_DIR="${OUTPUT_ROOT}/${SUBDIR}"
                      mkdir -p "${EXP_DIR}"

                      
                      LOG_FILE="${EXP_DIR}/train_log.txt"
                      MODEL_DIR="${EXP_DIR}/models"
                      RESULTS_DIR="${EXP_DIR}/results"
                      mkdir -p "${MODEL_DIR}" "${RESULTS_DIR}"

                      echo "=============================================="
                      echo " Starting experiment (data dir: ${DATA_DIR}):"
                      echo "   LR              = ${LR}"
                      echo "   BatchSize       = ${BS}"
                      echo "   MaxSeq          = ${MS}"
                      echo "   HiddenDim       = ${HD}"
                      echo "   w_ce            = ${WCE}"
                      echo "   w_dice          = ${WD}"
                      echo "   w_jaccard       = ${WJ}"
                      echo "   w_shape_mono    = ${WSM}"
                      echo "   w_depth_mono    = ${WDM}"
                      echo "   w_expand        = ${WE}"
                      echo "   DataDir         = ${DATA_DIR}"
                      echo "   OutputDir       = ${EXP_DIR}"
                      echo "=============================================="
                      echo ""

                     
                      ${PYTHON_EXEC} "${TRAIN_SCRIPT}" train \
                        --data_dir        "${DATA_DIR}" \
                        --epochs          "${EPOCHS}" \
                        --batch           "${BS}" \
                        --lr              "${LR}" \
                        --max_seq         "${MS}" \
                        --hidden_dim      "${HD}" \
                        --w_ce            "${WCE}" \
                        --w_dice          "${WD}" \
                        --w_jaccard       "${WJ}" \
                        --w_shape_mono    "${WSM}" \
                        --w_depth_mono    "${WDM}" \
                        --w_expand        "${WE}" \
                        --results_dir     "${RESULTS_DIR}" \
                        --model_out_dir   "${MODEL_DIR}" \
                        --log_out         "${LOG_FILE}" \
                        2>&1 | tee "${LOG_FILE}"

                      echo ""
                      echo " Finished: ${SUBDIR} (data dir: ${DATA_DIR})"
                      echo ""

                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done

  echo "All experiments for ${DATA_DIR} completed. Results under ${OUTPUT_ROOT}/"
  echo ""
done

echo "All data directories done!" 
