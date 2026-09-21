import torch
import torch.nn as nn


class TwoLayerEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x):
        return self.net(x)


class EdgeCorrection(nn.Module):
    def __init__(self, representation_dim=32, horizon_dim=8):
        super().__init__()
        input_dim = representation_dim * 2 + 2 + horizon_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class HydroToxNet(nn.Module):
    def __init__(
        self,
        ecological_dim,
        river_dim,
        meteorological_dim,
        representation_dim=32,
        horizon_dim=8,
        dropout=0.2,
        use_ecological=True,
        use_toxin_history=True,
        use_spatial_external=True,
        use_physical_prior=True,
        use_edge_correction=True,
    ):
        super().__init__()
        self.representation_dim = representation_dim
        self.use_ecological = use_ecological
        self.use_toxin_history = use_toxin_history
        self.use_spatial_external = use_spatial_external
        self.use_physical_prior = use_physical_prior
        self.use_edge_correction = use_edge_correction

        self.ecological_encoder = TwoLayerEncoder(
            ecological_dim,
            representation_dim,
            dropout,
        )

        self.toxin_encoder = TwoLayerEncoder(
            4,
            representation_dim,
            dropout,
        )

        self.horizon_embedding = nn.Embedding(3, horizon_dim)
        self.edge_correction = EdgeCorrection(representation_dim, horizon_dim)

        self.spatial_encoder = TwoLayerEncoder(
            representation_dim + river_dim + meteorological_dim,
            representation_dim,
            dropout,
        )

        fusion_dim = representation_dim * 5 + horizon_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        self.output_heads = nn.ModuleList([
            nn.Linear(32, 2),
            nn.Linear(32, 2),
            nn.Linear(32, 2),
        ])

    def physical_prior(self, distance_km, wind_alignment, source_mc_raw):
        distance_term = torch.exp(-distance_km / 10.0)
        direction_term = torch.clamp(wind_alignment, min=0.0)
        toxin_term = source_mc_raw / (source_mc_raw + 1.0)
        toxin_term = torch.clamp(toxin_term, min=0.0)
        return distance_term * direction_term * toxin_term

    def spatial_representation(
        self,
        source_repr,
        target_repr,
        source_mc_raw,
        source_mask,
        distance_km,
        normalized_distance,
        wind_alignment,
        horizon_embedding,
        river,
        meteorology,
    ):
        batch_size, n_stations, _ = source_repr.shape

        target_expand = target_repr[:, None, :].expand(-1, n_stations, -1)
        horizon_expand = horizon_embedding[:, None, :].expand(-1, n_stations, -1)

        edge_input = torch.cat(
            [
                source_repr,
                target_expand,
                normalized_distance.unsqueeze(-1),
                wind_alignment.unsqueeze(-1),
                horizon_expand,
            ],
            dim=-1,
        )

        if self.use_edge_correction:
            correction = self.edge_correction(edge_input)
        else:
            correction = torch.zeros_like(normalized_distance)

        if self.use_physical_prior:
            prior = self.physical_prior(
                distance_km,
                wind_alignment,
                source_mc_raw,
            )
            logits = torch.log(prior + 1e-6) + correction
        else:
            logits = correction

        logits = logits.masked_fill(~source_mask, -1e9)
        weights = torch.softmax(logits, dim=1)
        weights = weights * source_mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)

        pooled = torch.sum(weights.unsqueeze(-1) * source_repr, dim=1)
        external = torch.cat([pooled, river, meteorology], dim=-1)
        return self.spatial_encoder(external)

    def forward(
        self,
        target_ecological,
        toxin_history,
        source_ecological,
        source_mc_raw,
        source_mask,
        distance_km,
        normalized_distance,
        wind_alignment,
        river,
        meteorology,
    ):
        batch_size = target_ecological.shape[0]
        device = target_ecological.device

        target_repr = self.ecological_encoder(target_ecological)
        source_repr = self.ecological_encoder(source_ecological)
        toxin_repr = self.toxin_encoder(toxin_history)

        if not self.use_ecological:
            target_repr = torch.zeros_like(target_repr)

        if not self.use_toxin_history:
            toxin_repr = torch.zeros_like(toxin_repr)

        all_logits = []

        for horizon_idx in range(3):
            h = torch.full(
                (batch_size,),
                horizon_idx,
                dtype=torch.long,
                device=device,
            )
            h_emb = self.horizon_embedding(h)

            if self.use_spatial_external:
                spatial_repr = self.spatial_representation(
                    source_repr,
                    target_repr,
                    source_mc_raw,
                    source_mask,
                    distance_km,
                    normalized_distance,
                    wind_alignment,
                    h_emb,
                    river,
                    meteorology,
                )
            else:
                spatial_repr = torch.zeros_like(target_repr)

            fusion_input = torch.cat(
                [
                    target_repr,
                    toxin_repr,
                    spatial_repr,
                    target_repr * toxin_repr,
                    target_repr * spatial_repr,
                    h_emb,
                ],
                dim=-1,
            )

            fused = self.fusion(fusion_input)
            logits = self.output_heads[horizon_idx](fused)
            all_logits.append(logits)

        return torch.stack(all_logits, dim=1)
