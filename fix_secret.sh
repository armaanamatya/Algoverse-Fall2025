#!/bin/bash
# Script to replace API key in git history

git filter-branch --force --index-filter '
    git checkout-index -f -a
    if [ -f our-pipeline/process_datasets.py ]; then
        sed -i "s/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/os.getenv(\"OPENAI_API_KEY\")/g" our-pipeline/process_datasets.py
        git add our-pipeline/process_datasets.py
    fi
' --prune-empty --tag-name-filter cat -- --all
