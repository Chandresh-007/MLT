#!/bin/sh
# MindfulTech – Container Entrypoint
# ===================================
# Remaps MLT_* env vars to AWS_* since Amplify forbids the AWS_ prefix.

[ -n "$MLT_ACCESS_KEY_ID" ]     && export AWS_ACCESS_KEY_ID="$MLT_ACCESS_KEY_ID"
[ -n "$MLT_SECRET_ACCESS_KEY" ] && export AWS_SECRET_ACCESS_KEY="$MLT_SECRET_ACCESS_KEY"
[ -n "$MLT_REGION" ]            && export AWS_REGION="$MLT_REGION"

exec "$@"
