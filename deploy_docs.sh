#!/bin/bash

# Configuration
# Load environment variables if .env exists
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Use S3_BUCKET from environment if available, otherwise fallback to placeholder
BUCKET="${S3_BUCKET:-your-docs-bucket-name}"
DOCS_DIR="docs"
RENDER_DIR="docs/_site"

echo "🚀 Starting documentation deployment..."

# 1. Render the site
echo "📦 Rendering Quarto site..."
quarto render $DOCS_DIR

if [ $? -eq 0 ]; then
    echo "✅ Site rendered successfully in $RENDER_DIR"
else
    echo "❌ Error rendering site. Make sure Quarto is installed."
    exit 1
fi

# 2. Sync to S3
if [ "$BUCKET" == "your-docs-bucket-name" ]; then
    echo "⚠️  BUCKET_NAME not configured correctly in .env or script."
    echo "Please set S3_BUCKET in your .env or edit this script."
    exit 1
fi

echo "📤 Syncing to S3 bucket: s3://$BUCKET/docs/"
# Sync the rendered folder to S3
# Note: Removed --acl public-read because modern buckets often disable ACLs.
# Use a Bucket Policy for public access instead.
if [ -d "$RENDER_DIR" ]; then
    aws s3 sync "$RENDER_DIR/" s3://$BUCKET/docs/ --delete
else
    echo "❌ Error: Render directory $RENDER_DIR not found."
    exit 1
fi

echo ""
echo "💡 To finish deployment, ensure your S3 bucket is configured for Static Website Hosting."
echo "💡 If you get 403 Forbidden, ensure your Bucket Policy allow public read on the 'docs/*' path."
echo "🎉 Documentation deployed to s3://$BUCKET/docs/index.html!"
