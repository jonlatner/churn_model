###########################
        corr = df_cat_corr.corr().iloc[1:,:-1].copy()

        # Generate annotation labels array (of the same size as the heatmap data)- filling cells you don't want to annotate with an empty string ''
        annot_labels = np.empty_like(corr, dtype=str)
        annot_mask = np.where(corr > .7,round(corr,2),
                                np.where(corr < -.7,round(corr,2,
                                ""))

        # Create a mask
        mask = np.triu(np.ones_like(df_cat_corr.corr(), dtype=bool))
        # adjust mask and df
        mask = mask[1:, :-1]

        plt.figure(figsize = (10,10))        # Size of the figure
        sns.heatmap(corr,
                        mask=mask,
                        annot=annot_mask,
                        cmap='Blues',
                        # annot = True, 
                        fmt='.2f',
                        vmin=-1, vmax=1)
        plt.tight_layout()
        plt.show()
        
        fig = plt.gcf()  # or by other means, like plt.subplots
        figsize = fig.get_size_inches()
        fig.set_size_inches(figsize * 1.5)  # scale current size by 1.5
        plt.savefig(os.path.join(main_dir,graphs,'multivaritate_corr_finaldf.pdf'),bbox_inches='tight')
