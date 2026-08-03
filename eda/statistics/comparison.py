from scipy.stats import (
    kruskal,
    chi2_contingency,
)


def comparison_test(
    table,
    variable_type,
):


    if variable_type == "continuous":

        groups = [
            x["value"].values
            for _, x in table.groupby("group")
        ]


        stat, pvalue = kruskal(
            *groups
        )


        return {
            "test": "Kruskal-Wallis",
            "pvalue": pvalue,
        }


    else:

        contingency = (

            table
            .pivot_table(
                index="group",
                columns="feature",
                values="observations",
                fill_value=0,
            )

        )


        stat, pvalue, _, _ = chi2_contingency(
            contingency
        )


        return {
            "test": "Chi-square",
            "pvalue": pvalue,
        }
