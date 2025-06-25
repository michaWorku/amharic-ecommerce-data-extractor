Model Interpretability
======================

Understanding *why* a model makes a particular prediction is as crucial as the prediction itself, especially in sensitive applications like financial decision-making. The `src/models/model_evaluator.py` module includes preliminary exploration of model interpretability using SHAP (SHapley Additive exPlanations) and LIME (Local Interpretable Model-agnostic Explanations).

## SHAP Explanations

SHAP values help us understand the contribution of each word (or token) to a specific prediction.

* **Visualization Placeholder (Force Plot):**
    .. image:: ../_static/shap_force_plot_placeholder.png
       :alt: SHAP Force Plot Placeholder
       :width: 800px
       :align: center

* **Visualization Placeholder (Waterfall Plot):**
    .. image:: ../_static/shap_waterfall_plot_placeholder.png
       :alt: SHAP Waterfall Plot Placeholder
       :width: 600px
       :align: center

* **Interpretation:**
    * Red segments indicate features (words) that push the prediction towards the positive class (the predicted entity label).
    * Blue segments indicate features that push the prediction away from the positive class.
    * The magnitude of the bar reflects the strength of the influence.
    * This helps in identifying which words are most influential in determining an entity, for instance, specific Amharic words often associated with ``PRICE`` or ``LOCATION``.

## LIME Explanations

LIME provides local explanations by perturbing the input and observing changes in the model's prediction. This helps to highlight which parts of the input text are most important for a particular prediction.

* **Visualization Placeholder (LIME HTML Output):**
    .. image:: ../_static/lime_explanation_example_1.png
       :alt: LIME Explanation Placeholder
       :width: 800px
       :align: center

* **Interpretation:**
    * The visualization shows the words that contribute most to the model predicting the label 'O' for the token 'Rechargeable'.
    * Green words contribute positively (increase probability of the label at the target token).
    * Red words contribute negatively (decrease probability of the label at the target token).
    * The intensity of the color indicates the strength of the contribution.
    * Analyze which words in the surrounding context most influence the model's decision for the target token's label.

    * LIME's visualization shows how the model locally weighs different words in the sentence to arrive at a particular label for a specific token, even for words that are not explicitly entities themselves but provide crucial context.

* **Visualization Placeholder (LIME HTML Output):**
    .. image:: ../_static/lime_explanation_example_2.png
       :alt: LIME Explanation Placeholder
       :width: 800px
       :align: center

* **Interpretation:**
    * The visualization shows the words that contribute most to the model predicting the label 'O' for the token 'ቦታዎች'.
    * Green words contribute positively (increase probability of the label at the target token).
    * Red words contribute negatively (decrease probability of the label at the target token).
    * The intensity of the color indicates the strength of the contribution.
    * Analyze which words in the surrounding context most influence the model's decision for the target token's label.
    * LIME's visualization shows how the model locally weighs different words in the sentence to arrive at a particular label for a specific token, even for words that are not explicitly entities themselves but provide crucial context.


## Challenges in Interpretability

The SHAP and LIME implementations faced challenges in precisely mapping explanations back to original Amharic tokens, particularly due to the subword tokenization used by transformer models and the complexity of Amharic script. While initial attempts provided insights, full-scale, production-ready interpretability for complex Amharic NER would require further refinement of token-to-explanation alignment. Visualizations are best viewed in interactive notebook environments.