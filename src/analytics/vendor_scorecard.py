# src/analytics/vendor_scorecard.py (Conceptual - ensure your actual file matches this idea)
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class VendorScorecardGenerator:
    def __init__(self):
        logger.info("VendorScorecardGenerator initialized.")

    def generate_scorecard(self, df_predictions: pd.DataFrame, df_preprocessed_meta: pd.DataFrame) -> pd.DataFrame:
        """
        Generates a vendor scorecard by combining NER predictions with original message metadata.

        Args:
            df_predictions (pd.DataFrame): DataFrame containing 'message_id', 'token', 'predicted_label'
                                           from the NER model's output.
            df_preprocessed_meta (pd.DataFrame): DataFrame containing 'message_id', 'views', 'message_date',
                                                 'channel_username' from the preprocessed data.

        Returns:
            pd.DataFrame: A DataFrame with calculated vendor scores and KPIs.
        """
        logger.info("Generating vendor scorecard...")

        # Step 1: Merge predictions with original message metadata
        # We need to map predicted entities back to original messages/channels
        # The df_predictions has 'message_id' which is key for joining
        # Also need 'channel_username', 'message_date', 'views' from df_preprocessed_meta
        # Let's group predictions by message_id first to reconstruct entities per message
        message_entities = {}
        for _, row in df_predictions.iterrows():
            msg_id = row['message_id']
            if msg_id not in message_entities:
                message_entities[msg_id] = {'products': [], 'prices': [], 'locations': [], 'contact_info': []}

            label = row['predicted_label']
            token = row['token']

            # Basic entity extraction (can be refined for multi-word entities)
            if 'PRODUCT' in label:
                message_entities[msg_id]['products'].append(token)
            elif 'PRICE' in label:
                message_entities[msg_id]['prices'].append(token)
            elif 'LOC' in label:
                message_entities[msg_id]['locations'].append(token)
            elif 'CONTACT_INFO' in label:
                message_entities[msg_id]['contact_info'].append(token)

        # Convert to a DataFrame for easier merging
        entities_df = pd.DataFrame([
            {'message_id': mid,
             'products': ', '.join(set(data['products'])), # Use set to get unique products
             'prices': ', '.join(set(data['prices'])),
             'locations': ', '.join(set(data['locations'])),
             'contact_info': ', '.join(set(data['contact_info']))
            } for mid, data in message_entities.items()
        ])

        # Merge with preprocessed metadata
        # IMPORTANT: Ensure 'message_id' is consistent and available in both DFs
        # df_preprocessed_meta should ideally have 'message_id' from the scraper
        merged_df = pd.merge(df_preprocessed_meta[['message_id', 'channel_username', 'message_date', 'views']],
                             entities_df,
                             on='message_id',
                             how='left')

        # Fill NaN for entity columns with empty strings
        for col in ['products', 'prices', 'locations', 'contact_info']:
            merged_df[col] = merged_df[col].fillna('')

        # Step 2: Calculate KPIs per vendor (channel_username)
        vendor_kpis = {}
        for channel_username in merged_df['channel_username'].unique():
            channel_df = merged_df[merged_df['channel_username'] == channel_username].copy()

            total_posts = len(channel_df)
            avg_views_per_post = channel_df['views'].mean() if not channel_df.empty else 0
            
            # Posts per week (assuming 'message_date' is datetime)
            if not channel_df.empty and channel_df['message_date'].min() and channel_df['message_date'].max():
                time_span_days = (channel_df['message_date'].max() - channel_df['message_date'].min()).days
                if time_span_days > 0:
                    posts_per_week = (total_posts / time_span_days) * 7
                else:
                    posts_per_week = total_posts # All posts on same day, count as total if only 1 week
            else:
                posts_per_week = 0

            # Example: Top product by views (simplified)
            top_product = "N/A"
            top_product_price = "N/A"
            # Assuming product and price are in `products` and `prices` columns
            # This needs more sophisticated parsing for real-world use if multiple items are in one string
            product_posts = channel_df[channel_df['products'].str.strip() != '']
            if not product_posts.empty:
                # For simplicity, let's take the product from the highest viewed post
                # In a real scenario, you'd parse `products` and `prices` into lists
                # and analyze them more deeply.
                highest_viewed_product_post = product_posts.loc[product_posts['views'].idxmax()]
                top_product = highest_viewed_product_post['products'].split(', ')[0] if highest_viewed_product_post['products'] else "N/A"
                top_product_price = highest_viewed_product_post['prices'].split(', ')[0] if highest_viewed_product_post['prices'] else "N/A"

            # Average price point from extracted prices (very basic, needs robust parsing)
            all_prices = []
            for prices_str in channel_df['prices']:
                if prices_str:
                    for p_val in re.findall(r'\d+\.?\d*', prices_str): # Extract numbers from price string
                        try:
                            all_prices.append(float(p_val))
                        except ValueError:
                            pass
            avg_price_point = np.mean(all_prices) if all_prices else 0

            # Store KPIs
            vendor_kpis[channel_username] = {
                'total_posts': total_posts,
                'avg_views_per_post': avg_views_per_post,
                'posts_per_week': posts_per_week,
                'avg_price_point': avg_price_point,
                'top_product': top_product,
                'top_product_price': top_product_price,
            }

        scorecard_df = pd.DataFrame.from_dict(vendor_kpis, orient='index')
        scorecard_df.index.name = 'channel_username'
        scorecard_df = scorecard_df.reset_index()

        # Step 3: Calculate composite Lending Score
        # Normalize KPIs (simple min-max scaling for demonstration)
        # Ensure that KPIs are not all zero before normalizing to avoid division by zero
        if not scorecard_df.empty:
            for col in ['total_posts', 'avg_views_per_post', 'posts_per_week', 'avg_price_point']:
                min_val = scorecard_df[col].min()
                max_val = scorecard_df[col].max()
                if max_val > min_val:
                    scorecard_df[f'normalized_{col}'] = (scorecard_df[col] - min_val) / (max_val - min_val)
                else:
                    scorecard_df[f'normalized_{col}'] = 0.0 # All values are same, normalized to 0

            # Define weights for the lending score
            # These weights can be adjusted based on business priorities
            weights = {
                'normalized_total_posts': 0.25,
                'normalized_avg_views_per_post': 0.35,
                'normalized_posts_per_week': 0.20,
                'normalized_avg_price_point': 0.20,
            }

            scorecard_df['lending_score'] = scorecard_df.apply(
                lambda row: sum(row[f'normalized_{k}'] * w for k, w in weights.items()),
                axis=1
            )
        else:
            scorecard_df['lending_score'] = 0.0


        # Sort by lending score
        scorecard_df = scorecard_df.sort_values(by='lending_score', ascending=False).reset_index(drop=True)

        logger.info("Vendor scorecard generated successfully.")
        return scorecard_df